import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from .attempts import Attempt, Outcome, Result
from .measure import Measurement, Target
from .tasks import FactCheck, FactStatus

STATE_TYPES = (Target, Measurement, FactCheck, FactStatus, Attempt, Outcome, Result)
DATABASE = "checkpoints.sqlite"


def serializer() -> JsonPlusSerializer:
    """
    Сериализатор состояния графа

    Описание:
        Восстанавливает из сохраненного состояния только типы ruTS-adapt
        из STATE_TYPES: тип не из списка LangGraph блокирует или восстанавливает
        не тем классом, поэтому новый тип в состоянии нужно добавить сюда

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
        yield SqliteSaver(connection, serde=serializer())
    finally:
        connection.close()
