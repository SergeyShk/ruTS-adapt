import argparse
import io
import re
import sys
from collections.abc import Iterable, Sequence
from typing import Any, NoReturn

from . import __version__

# Сообщения argparse, которые может получить команда; остальные печатаются как есть
ARGPARSE_ERRORS = (
    (re.compile(r"unrecognized arguments: (.*)"), "неизвестные аргументы: {0}"),
    (
        re.compile(r"argument (\S+): ignored explicit argument (.*)"),
        "аргумент {0}: лишнее значение {1}",
    ),
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
        Без аргументов печатает справку

    Аргументы:
        argv (list[str]): Аргументы командной строки; None - sys.argv
    """
    # Справка и ошибки по-русски не должны падать на консоли с однобайтной кодировкой
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="replace")
    parser = ArgumentParser(
        prog="ruts-adapt",
        description="Адаптация русского текста под уровень читателя: модель переписывает текст, "
        "ruTS измеряет его уровень, LLM-судья проверяет, что смысл сохранен",
    )
    parser.options.add_argument(
        "--version",
        action="version",
        version=f"ruts-adapt {__version__}",
        help="показать версию и выйти",
    )
    parser.parse_args(argv)
    parser.print_help()
