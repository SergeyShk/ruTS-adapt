import argparse
import io
import os
import re
import signal
import sqlite3
import sys
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack, suppress
from importlib.metadata import version
from pathlib import Path
from types import FrameType
from typing import Any, NoReturn

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot
from pydantic import ValidationError

from . import __version__
from .attempts import Outcome, Result
from .checkpoint import STATE_VERSION, ThreadBusyError, checkpointer, prune, thread_lock
from .graph import Context, State, build_graph
from .language import check_russian
from .measure import MAX_GRADE, MIN_GRADE, Target
from .model import ModelError, chat_model
from .report import Reporter
from .settings import Settings

NOT_ACCEPTED = 3
PAUSED = 4
INTERRUPTED = 130
MAX_ATTEMPTS = 10
DEFAULT_ATTEMPTS = 3
# С «-» в начале имя в подсказке читалось бы как параметр
THREAD = re.compile(r"\w[\w.-]*")
FILE_ERRORS = (
    (FileNotFoundError, "файл не найден"),
    (PermissionError, "нет доступа"),
    (UnicodeDecodeError, "текст не в кодировке UTF-8"),
)

# Сообщения argparse, которые может получить команда; остальные печатаются как есть
ARGPARSE_ERRORS = (
    (re.compile(r"unrecognized arguments: (.*)"), "неизвестные аргументы: {0}"),
    (re.compile(r"the following arguments are required: (.*)"), "нужны аргументы: {0}"),
    (
        re.compile(r"argument (\S+): ignored explicit argument (.*)"),
        "аргумент {0}: лишнее значение {1}",
    ),
    (re.compile(r"argument (\S+): expected one argument"), "аргумент {0}: нужно значение"),
    (
        re.compile(r"argument (\S+): not allowed with argument (\S+)"),
        "аргумент {0}: нельзя вместе с {1}",
    ),
    (re.compile(r"argument (\S+): (.*)"), "аргумент {0}: {1}"),
)


class HelpFormatter(argparse.HelpFormatter):
    """Справка с русской строкой использования"""

    def add_usage(
        self,
        usage: str | None,
        actions: Iterable[argparse.Action],
        groups: Iterable[Any],
        prefix: str | None = None,
    ) -> None:
        super().add_usage(usage, actions, groups, "использование: " if prefix is None else prefix)


class ArgumentParser(argparse.ArgumentParser):
    """Разбор аргументов с русскими справкой и сообщениями об ошибках"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(formatter_class=HelpFormatter, add_help=False, **kwargs)
        self.arguments = self.add_argument_group("аргументы")
        options = self.add_argument_group("параметры")
        options.add_argument("-h", "--help", action="help", help="показать эту справку и выйти")
        self.options = options

    def error(self, message: str) -> NoReturn:
        for pattern, template in ARGPARSE_ERRORS:
            if match := pattern.fullmatch(message):
                message = template.format(*match.groups())
                break
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: ошибка: {message}\n")


def main(argv: Sequence[str] | None = None) -> None:
    """
    Запуск команды ruts-adapt

    Описание:
        Новый поток адаптации читает текст из файла или stdin, адаптирует его
        к классу читателя и печатает итоговый текст в stdout, а ход работы, таблицу
        попыток и итог - в stderr. Состояние потока сохраняется после каждого шага:
        --thread продолжает поток после остановки или передает решение по кандидату
        на итог, если поток запущен с --review. Код 0 - текст на уровне цели,
        NOT_ACCEPTED - лучшая попытка не прошла проверку (текст все равно
        в stdout), PAUSED - кандидат ждет решения; неверные аргументы, настройки,
        входной текст и занятый другим процессом поток - код 2, ошибки модели -
        код 1, Ctrl+C - код 130 без трейсбека, не дожидаясь ответа модели

    Аргументы:
        argv (list[str]): Аргументы командной строки; None - sys.argv
    """
    # Справка и ошибки по-русски не должны падать на консоли с однобайтной кодировкой
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="replace")
            # В файл или канал - в UTF-8, как и вход, а не в кодировке консоли Windows
            if not stream.isatty():
                stream.reconfigure(encoding="utf-8")
    parser = ArgumentParser(
        prog="ruts-adapt",
        usage="%(prog)s [-h] -g КЛАСС [-n N] [-r] [-t ПОТОК] [ФАЙЛ]\n"
        "       %(prog)s -t ПОТОК [--approve | --reject ЗАМЕЧАНИЕ]",
        description="Адаптация русского текста под уровень читателя: модель переписывает текст, "
        "ruTS измеряет его уровень, LLM-судья проверяет, что факты сохранены. Итоговый текст "
        "печатается в stdout, ход работы и таблица попыток - в stderr. Состояние сохраняется "
        "после каждого шага: остановленный поток продолжается по --thread",
        epilog="Модели задаются переменными окружения RUTS_ADAPT_WRITER_MODEL "
        "и RUTS_ADAPT_JUDGE_MODEL в виде провайдер:модель (anthropic, openai, google_genai, "
        "ollama), по умолчанию anthropic:claude-opus-5-5; усилие и лимит ответа - "
        "RUTS_ADAPT_<WRITER|JUDGE>_EFFORT и _MAX_TOKENS, окно контекста локальной модели - "
        "_CONTEXT, каталог данных с потоками - RUTS_ADAPT_DATA_DIR. Ключи API - переменные "
        "провайдеров: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY. Коды возврата: 0 - текст "
        f"на уровне цели, {NOT_ACCEPTED} - лучшая попытка не прошла проверку (текст все равно "
        f"в stdout), {PAUSED} - кандидат ждет решения, 1 - ошибка модели, 2 - неверные "
        "аргументы, настройки или текст, 130 - Ctrl+C",
    )
    parser.arguments.add_argument(
        "file",
        nargs="?",
        metavar="ФАЙЛ",
        help="файл с текстом в UTF-8; без него или с - текст читается из stdin",
    )
    parser.options.add_argument(
        "-g",
        "--grade",
        type=_target,
        metavar="КЛАСС",
        help=f"класс читателя: 1-11 - класс школы, 12-{MAX_GRADE} - год обучения в вузе; "
        "цель - его ступень: 1-3, 4-6, 7-9, 10-11-й класс, 1-3 или 4-6-й курс",
    )
    parser.options.add_argument(
        "-n",
        "--attempts",
        type=_attempts,
        metavar="N",
        help=f"наибольшее число попыток переписать текст, от 1 до {MAX_ATTEMPTS}; по умолчанию "
        f"{DEFAULT_ATTEMPTS}",
    )
    parser.options.add_argument(
        "-r",
        "--review",
        action="store_true",
        help="остановиться перед итогом и ждать решения: --approve или --reject",
    )
    parser.options.add_argument(
        "-t",
        "--thread",
        type=_thread,
        metavar="ПОТОК",
        help="поток адаптации: новый с этим именем или продолжение сохраненного",
    )
    decision = parser.options.add_mutually_exclusive_group()
    decision.add_argument(
        "--approve", action="store_true", help="принять кандидата на итог в потоке с --review"
    )
    decision.add_argument(
        "--reject",
        metavar="ЗАМЕЧАНИЕ",
        help="вернуть кандидата с замечанием: писатель перепишет текст с его учетом",
    )
    parser.options.add_argument(
        "--version",
        action="version",
        version=f"ruts-adapt {__version__}, ruts {version('ruts')}",
        help="показать версии ruts-adapt и ruts и выйти",
    )
    args = parser.parse_args(argv)
    try:
        _run(parser, args)
    except KeyboardInterrupt:
        sys.exit(INTERRUPTED)
    except BrokenPipeError:
        # Канал закрыт читателем (| head): остаток вывода никому не нужен, трейсбек тоже
        devnull = os.open(os.devnull, os.O_WRONLY)
        for stream in (sys.stdout, sys.stderr):
            os.dup2(devnull, stream.fileno())
        sys.exit(1)


def _run(parser: ArgumentParser, args: argparse.Namespace) -> None:
    """Новый поток или продолжение сохраненного, адаптация и печать итога"""
    if args.reject is not None and not args.reject.strip():
        parser.error("замечание --reject не должно быть пустым")
    decided = args.approve or args.reject is not None
    if args.thread is None and decided:
        parser.error("решение передается сохраненному потоку: нужен --thread")
    try:
        settings = Settings.from_env()
    except ValueError as error:
        parser.error(str(error))
    # Без --thread поток новый: аргументы и текст проверяются до открытия хранилища
    started = _start(parser, args) if args.thread is None else None
    thread = args.thread or uuid.uuid4().hex[:8]
    config: RunnableConfig = {"configurable": {"thread_id": thread}}
    with ExitStack() as stack:
        try:
            stack.enter_context(thread_lock(settings.data_dir, thread))
            saver = stack.enter_context(checkpointer(settings.data_dir))
        except ThreadBusyError as error:
            parser.error(str(error))
        except (OSError, sqlite3.Error) as error:
            parser.error(f"не удалось открыть хранилище потоков в {settings.data_dir}: {error}")
        graph = build_graph(checkpointer=saver)
        try:
            snapshot = graph.get_state(config)
        except sqlite3.Error as error:
            parser.error(f"не удалось открыть хранилище потоков в {settings.data_dir}: {error}")
        except (ValueError, TypeError):
            parser.error(f"поток {thread} поврежден: начните новый поток с другим именем")
        graph_input: State | Command[Any] | None
        values: Mapping[str, Any]
        if snapshot.values:
            _check_saved(parser, args, thread, snapshot)
            values = snapshot.values
            if not snapshot.next:
                print(f"Поток {thread} уже завершен", file=sys.stderr)
                _exit_with(values["result"])
            if snapshot.interrupts and not decided:
                reporter = Reporter(values["target"], values["max_attempts"], sys.stderr)
                reporter.restore(values)
                reporter.pause(thread, snapshot.interrupts[0].value)
                sys.exit(PAUSED)
            graph_input = Command(resume={"feedback": args.reject}) if decided else None
        else:
            if decided or args.grade is None:
                parser.error(
                    f"поток {thread} не найден в {settings.data_dir}"
                    + ("" if decided else "; новый поток - с -g КЛАСС")
                )
            started = started or _start(parser, args)
            graph_input, values = started, started
        # Принятому кандидату модель не нужна: граф сразу идет к итогу
        context = None if args.approve else _context(parser, settings)
        if snapshot.values:
            print(f"Продолжаем поток {thread}", file=sys.stderr)
        else:
            print(
                f"Поток {thread}: продолжить после остановки - ruts-adapt --thread {thread}",
                file=sys.stderr,
            )
        reporter = Reporter(values["target"], values["max_attempts"], sys.stderr)
        reporter.restore(values)
        try:
            result = adapt(graph, graph_input, config, context, reporter)
        except ModelError as error:
            _fail(f"ошибка модели: {error}. Продолжить: ruts-adapt --thread {thread}")
        except sqlite3.Error as error:
            _fail(f"не удалось сохранить поток в {settings.data_dir}: {error}")
        # Неудавшаяся чистка не мешает итогу: старые чекпоинты удалит следующий запуск
        with suppress(sqlite3.Error):
            prune(saver, thread)
    if result is None:
        sys.exit(PAUSED)
    _exit_with(result)


def _context(parser: ArgumentParser, settings: Settings) -> Context:
    """Модели писателя и судьи; ошибка настроек или ключа - до первого шага потока"""
    try:
        return Context(writer=chat_model(settings.writer), judge=chat_model(settings.judge))
    except ValidationError as error:
        reasons = "; ".join(str(item["msg"]) for item in error.errors())
        parser.error(f"не удалось создать модель: {reasons}")
    except ValueError as error:
        parser.error(str(error))
    except Exception as error:
        _fail(f"не удалось создать модель: {error}")


def _exit_with(result: Result) -> NoReturn:
    """Итоговый текст в stdout и код: 0 - на уровне цели, NOT_ACCEPTED - лучшая непринятая"""
    print(result.text.rstrip("\n"))
    sys.exit(NOT_ACCEPTED if result.outcome is Outcome.BEST else 0)


def _start(parser: ArgumentParser, args: argparse.Namespace) -> State:
    """Вход графа для нового потока"""
    if args.grade is None:
        parser.error("нужен аргумент -g/--grade")
    try:
        text = _read(args.file or "-")
        check_russian(text)
    except ValueError as error:
        parser.error(str(error))
    return {
        "source": text,
        "target": args.grade,
        "max_attempts": args.attempts or DEFAULT_ATTEMPTS,
        "review": args.review,
        "version": STATE_VERSION,
    }


def _check_saved(
    parser: ArgumentParser, args: argparse.Namespace, thread: str, snapshot: StateSnapshot
) -> None:
    """Сохраненный поток можно продолжить этими аргументами: версия, решение, без новых данных"""
    if snapshot.values.get("version") != STATE_VERSION:
        parser.error(
            f"поток {thread} сохранен другой версией ruts-adapt: начните новый поток "
            "с другим именем"
        )
    given = [
        name
        for name, value in (
            ("ФАЙЛ", args.file),
            ("--grade", args.grade),
            ("--attempts", args.attempts),
            ("--review", args.review or None),
        )
        if value is not None
    ]
    if given:
        parser.error(f"поток {thread} уже есть, продолжение не принимает: {', '.join(given)}")
    decided = args.approve or args.reject is not None
    if decided and not snapshot.next:
        parser.error(f"поток {thread} уже завершен")
    if decided and not snapshot.interrupts:
        parser.error(f"поток {thread} не ждет решения")


def adapt(
    graph: CompiledStateGraph[State, Context],
    graph_input: State | Command[Any] | None,
    config: RunnableConfig,
    context: Context | None,
    reporter: Reporter,
) -> Result | None:
    """
    Адаптация текста графом с печатью хода работы

    Описание:
        Состояние сохраняется после каждого шага до перехода к следующему, чтобы
        поток продолжился с места остановки, даже если процесс прерван

    Аргументы:
        graph (CompiledStateGraph): Граф с чекпоинтером
        graph_input (State | Command | None): Вход нового потока, решение по кандидату
            или None - продолжение сохраненного потока с места остановки
        config (RunnableConfig): Конфигурация с потоком (thread_id)
        context (Context | None): Модели писателя и судьи; None, если граф их не вызовет
        reporter (Reporter): Печать хода работы

    Вывод:
        Result | None: Итог адаптации; None, если кандидат на итог ждет решения

    Исключения:
        ModelError: Если ответ модели нельзя использовать и после повторов
    """
    result: Result | None = None
    pause: Mapping[str, Any] | None = None
    previous = signal.signal(signal.SIGINT, _interrupt)
    try:
        with get_usage_metadata_callback() as usage:
            for update in graph.stream(
                graph_input, config, context=context, stream_mode="updates", durability="sync"
            ):
                for node, values in update.items():
                    if node == "__interrupt__":
                        pause = values[0].value
                        continue
                    reporter.update(node, values)
                    if node == "finish":
                        result = values["result"]
    finally:
        signal.signal(signal.SIGINT, previous)
    tokens = sum(
        item["input_tokens"] + item["output_tokens"] for item in usage.usage_metadata.values()
    )
    thread = config["configurable"]["thread_id"]
    if pause is not None:
        reporter.pause(thread, pause, tokens)
        return None
    assert result is not None
    reporter.finish(result, tokens)
    return result


def _interrupt(signum: int, frame: FrameType | None) -> None:
    """Ctrl+C: выход сразу, не дожидаясь вызова модели в параллельной ветви графа"""
    # KeyboardInterrupt дождался бы потоков LangGraph, а запрос к модели идет минутами
    sys.stderr.flush()
    os._exit(INTERRUPTED)


def _target(raw: str) -> Target:
    """Цель по аргументу --grade"""
    try:
        return Target.from_grade(int(raw))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"нужен класс от {MIN_GRADE} до {MAX_GRADE}, получено {raw!r}"
        ) from None


def _thread(raw: str) -> str:
    """Имя потока по аргументу --thread: его можно скопировать в команду без кавычек"""
    if not THREAD.fullmatch(raw):
        raise argparse.ArgumentTypeError(
            f"имя потока - буквы, цифры, «-», «_» и «.», начиная с буквы или цифры, "
            f"получено {raw!r}"
        )
    return raw


def _attempts(raw: str) -> int:
    """Число попыток по аргументу --attempts"""
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if not 1 <= value <= MAX_ATTEMPTS:
        raise argparse.ArgumentTypeError(
            f"нужно целое число от 1 до {MAX_ATTEMPTS}, получено {raw!r}"
        )
    return value


def _read(file: str) -> str:
    """
    Текст из файла или stdin

    Исключения:
        ValueError: Если текст не прочитать или stdin - консоль без текста
    """
    if file == "-":
        if sys.stdin.isatty():
            raise ValueError("нет текста: укажите ФАЙЛ или передайте текст через stdin")
        data = sys.stdin.buffer.read()
        try:
            # Как при чтении файла: без BOM и с переводами строк \n
            return io.TextIOWrapper(io.BytesIO(data), encoding="utf-8-sig").read()
        except UnicodeDecodeError:
            raise ValueError("текст в stdin не в кодировке UTF-8") from None
    path = Path(file)
    # Каталог на Windows не читается с PermissionError, а не IsADirectoryError
    if path.is_dir():
        raise ValueError(f"не удалось прочитать {file}: это каталог")
    try:
        return path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        reason = next((text for kind, text in FILE_ERRORS if isinstance(error, kind)), str(error))
        raise ValueError(f"не удалось прочитать {file}: {reason}") from None


def _fail(message: str) -> NoReturn:
    """Ошибка выполнения: сообщение в stderr и код 1"""
    print(f"ruts-adapt: {message}", file=sys.stderr)
    sys.exit(1)
