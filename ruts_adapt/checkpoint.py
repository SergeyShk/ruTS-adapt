import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from .attempts import Attempt, Outcome, Result
from .measure import Measurement, Target
from .tasks import FactCheck, FactStatus

STATE_TYPES = (Target, Measurement, FactCheck, FactStatus, Attempt, Outcome, Result)
# Увеличивать при изменении полей State и классов из STATE_TYPES: старые потоки так не восстановить
STATE_VERSION = 2
DATABASE = "checkpoints.sqlite"
LOCKS = "locks"
# Секунды ожидания, пока другой процесс создает таблицы новой базы
SETUP_TIMEOUT = 5.0


class ThreadBusyError(Exception):
    """Поток уже открыт другим процессом"""


def serializer() -> JsonPlusSerializer:
    """
    Сериализатор состояния графа

    Описание:
        Восстанавливает из сохраненного состояния только типы ruTS-adapt
        из STATE_TYPES: тип не из списка LangGraph блокирует или восстанавливает
        не тем классом, поэтому новый тип в состоянии нужно добавить сюда. Объект,
        класс которого с тех пор изменился, восстанавливается как None, поэтому
        поток хранит STATE_VERSION, а продолжение сверяет ее

    Вывод:
        JsonPlusSerializer: Сериализатор со списком разрешенных типов
    """
    return JsonPlusSerializer(
        allowed_msgpack_modules=[(kind.__module__, kind.__name__) for kind in STATE_TYPES]
    )


@contextmanager
def checkpointer(data_dir: Path) -> Iterator[SqliteSaver]:
    """
    Хранилище состояния графа в SQLite

    Описание:
        База DATABASE в каталоге данных, каталог создается при необходимости.
        Состояние каждого потока адаптации сохраняется после каждого шага графа

    Аргументы:
        data_dir (Path): Каталог данных

    Вывод:
        SqliteSaver: Хранилище, пока открыт контекст
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    # LangGraph пишет состояние и из потоков параллельных ветвей
    connection = sqlite3.connect(data_dir / DATABASE, check_same_thread=False)
    try:
        saver = SqliteSaver(connection, serde=serializer())
        _setup(saver)
        yield saver
    finally:
        connection.close()


@contextmanager
def thread_lock(data_dir: Path, thread: str) -> Iterator[None]:
    """
    Исключительная блокировка потока, пока открыт контекст

    Описание:
        Блокировка - открытая транзакция в файле потока в каталоге LOCKS: другой
        процесс не может работать с тем же потоком, а завершение процесса,
        в том числе аварийное, блокировку снимает

    Аргументы:
        data_dir (Path): Каталог данных
        thread (str): Поток адаптации

    Исключения:
        ThreadBusyError: Если поток открыт другим процессом
    """
    path = data_dir / LOCKS / f"{thread}.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=0, isolation_level=None)
    try:
        try:
            connection.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError:
            raise ThreadBusyError(f"поток {thread} уже выполняется в другом процессе") from None
        yield
    finally:
        connection.close()


def prune(saver: SqliteSaver, thread: str) -> None:
    """
    Удаление всех чекпоинтов потока, кроме последнего

    Описание:
        Продолжению потока нужен только последний чекпоинт с его записями,
        а каждый чекпоинт хранит состояние целиком - с исходником и всеми
        попытками, так что без чистки база растет квадратично по числу попыток

    Аргументы:
        saver (SqliteSaver): Хранилище
        thread (str): Поток адаптации
    """
    latest = saver.get_tuple({"configurable": {"thread_id": thread}})
    if latest is None:
        return
    checkpoint_id = latest.config["configurable"]["checkpoint_id"]
    with saver.lock, saver.conn:
        for table in ("checkpoints", "writes"):
            saver.conn.execute(
                f"DELETE FROM {table} WHERE thread_id = ? AND checkpoint_id != ?",
                (thread, checkpoint_id),
            )


def _setup(saver: SqliteSaver) -> None:
    """Таблицы базы; если их одновременно создает другой процесс - повтор до SETUP_TIMEOUT"""
    deadline = time.monotonic() + SETUP_TIMEOUT
    while True:
        try:
            saver.setup()
            return
        except sqlite3.OperationalError as error:
            if "locked" not in str(error) or time.monotonic() >= deadline:
                raise
            time.sleep(0.05)
