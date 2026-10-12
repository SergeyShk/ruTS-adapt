import argparse
import io
import os
import re
import signal
import sys
import uuid
from collections.abc import Iterable, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from types import FrameType
from typing import Any, NoReturn

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot

from . import __version__
from .attempts import Result
from .checkpoint import checkpointer
from .graph import Context, State, build_graph
from .language import check_russian
from .measure import MAX_GRADE, MIN_GRADE, Target
from .model import ModelError, chat_model
from .report import Reporter
from .settings import Settings

INTERRUPTED = 130
MAX_ATTEMPTS = 10
DEFAULT_ATTEMPTS = 3
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
        на итог, если поток запущен с --review. Неверные аргументы, настройки
        и входной текст завершают команду с кодом 2, ошибки модели - с кодом 1,
        Ctrl+C - с кодом 130 без трейсбека, не дожидаясь ответа модели

    Аргументы:
        argv (list[str]): Аргументы командной строки; None - sys.argv
    """
    # Справка и ошибки по-русски не должны падать на консоли с однобайтной кодировкой
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="replace")
    # Итоговый текст в файл или канал - в UTF-8, как и вход, а не в кодировке консоли Windows
    if isinstance(sys.stdout, io.TextIOWrapper) and not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8")
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
        "провайдеров: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY",
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


def _run(parser: ArgumentParser, args: argparse.Namespace) -> None:
    """Новый поток или продолжение сохраненного, адаптация и печать итога"""
    if args.reject is not None and not args.reject.strip():
        parser.error("замечание --reject не должно быть пустым")
    try:
        settings = Settings.from_env()
    except ValueError as error:
        parser.error(str(error))
    with checkpointer(settings.data_dir) as saver:
        graph = build_graph(checkpointer=saver)
        thread = args.thread or uuid.uuid4().hex[:8]
        config: RunnableConfig = {"configurable": {"thread_id": thread}}
        snapshot = graph.get_state(config)
        graph_input: State | Command[Any] | None
        values: Mapping[str, Any]
        if snapshot.values:
            graph_input = _resume(parser, args, thread, snapshot)
            if not snapshot.next:
                print(snapshot.values["result"].text)
                print(f"Поток {thread} уже завершен", file=sys.stderr)
                return
            values = snapshot.values
        else:
            started = _start(parser, args, thread)
            graph_input, values = started, started
        try:
            context = Context(writer=chat_model(settings.writer), judge=chat_model(settings.judge))
        except ValueError as error:
            parser.error(str(error))
        except Exception as error:
            _fail(f"не удалось создать модель: {error}")
        reporter = Reporter(values["target"], values["max_attempts"], sys.stderr)
        reporter.restore(values)
        try:
            result = adapt(graph, graph_input, config, context, reporter)
        except ModelError as error:
            _fail(f"ошибка модели: {error}. Продолжить: ruts-adapt --thread {thread}")
    if result is not None:
        print(result.text)


def _start(parser: ArgumentParser, args: argparse.Namespace, thread: str) -> State:
    """Вход графа для нового потока"""
    if args.approve or args.reject is not None:
        parser.error(f"поток {thread} не найден")
    if args.grade is None:
        parser.error("нужен аргумент -g/--grade")
    try:
        text = _read(args.file or "-")
        check_russian(text)
    except ValueError as error:
        parser.error(str(error))
    print(
        f"Поток {thread}: продолжить после остановки - ruts-adapt --thread {thread}",
        file=sys.stderr,
    )
    return {
        "source": text,
        "target": args.grade,
        "max_attempts": args.attempts or DEFAULT_ATTEMPTS,
        "review": args.review,
    }


def _resume(
    parser: ArgumentParser, args: argparse.Namespace, thread: str, snapshot: StateSnapshot
) -> Command[Any] | None:
    """
    Вход графа для сохраненного потока

    Вывод:
        Command | None: Решение по кандидату или None - продолжение с места остановки
    """
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
    if not snapshot.next:
        if decided:
            parser.error(f"поток {thread} уже завершен")
        return None
    if snapshot.interrupts and not decided:
        parser.error(f"поток {thread} ждет решения: --approve или --reject ЗАМЕЧАНИЕ")
    if decided and not snapshot.interrupts:
        parser.error(f"поток {thread} не ждет решения")
    print(f"Продолжаем поток {thread}", file=sys.stderr)
    return Command(resume={"feedback": args.reject}) if decided else None


def adapt(
    graph: CompiledStateGraph[State, Context],
    graph_input: State | Command[Any] | None,
    config: RunnableConfig,
    context: Context,
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
        context (Context): Модели писателя и судьи
        reporter (Reporter): Печать хода работы

    Вывод:
        Result | None: Итог адаптации; None, если кандидат на итог ждет решения

    Исключения:
        ModelError: Если ответ модели нельзя использовать и после повторов
    """
    result: Result | None = None
    paused = False
    previous = signal.signal(signal.SIGINT, _interrupt)
    try:
        with get_usage_metadata_callback() as usage:
            for update in graph.stream(
                graph_input, config, context=context, stream_mode="updates", durability="sync"
            ):
                for node, values in update.items():
                    if node == "__interrupt__":
                        paused = True
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
    if paused:
        reporter.pause(thread, tokens)
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
            return data.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("текст в stdin не в кодировке UTF-8") from None
    path = Path(file)
    # Каталог на Windows не читается с PermissionError, а не IsADirectoryError
    if path.is_dir():
        raise ValueError(f"не удалось прочитать {file}: это каталог")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        reason = next((text for kind, text in FILE_ERRORS if isinstance(error, kind)), str(error))
        raise ValueError(f"не удалось прочитать {file}: {reason}") from None


def _fail(message: str) -> NoReturn:
    """Ошибка выполнения: сообщение в stderr и код 1"""
    print(f"ruts-adapt: {message}", file=sys.stderr)
    sys.exit(1)
