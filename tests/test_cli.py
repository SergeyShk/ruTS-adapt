import importlib
import io
import os
import runpy
import signal
import sys
from importlib.metadata import version
from types import SimpleNamespace

import pytest

import ruts_adapt
from ruts_adapt import cli
from ruts_adapt.cli import ArgumentParser, main
from ruts_adapt.graph import build_graph
from ruts_adapt.tasks import Draft, Facts, Verdict
from tests.fakes import FakeChatModel, anthropic
from tests.samples import DECREE, FACTS, KEPT, SIMPLE
from tests.test_graph import FAST

VERSION = f"ruts-adapt {ruts_adapt.__version__}, ruts {version('ruts')}\n"


class Stdin(io.TextIOWrapper):
    """Ввод, который выглядит как консоль или как канал"""

    def __init__(self, data: bytes, tty: bool = False) -> None:
        super().__init__(io.BytesIO(data), encoding="utf-8")
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


@pytest.fixture
def stdin(monkeypatch):
    def set_stdin(data: bytes, tty: bool = False) -> None:
        monkeypatch.setattr(sys, "stdin", Stdin(data, tty))

    return set_stdin


@pytest.fixture
def models(monkeypatch):
    """Подмена моделей: писатель создается первым, судья - вторым"""

    def set_models(writer: FakeChatModel, judge: FakeChatModel) -> None:
        queue = iter([writer, judge])
        monkeypatch.setattr(cli, "chat_model", lambda settings: next(queue))

    return set_models


def fakes():
    writer = FakeChatModel(replies=[anthropic(Draft(text=SIMPLE))])
    judge = FakeChatModel(replies=[anthropic(Facts(facts=FACTS)), anthropic(KEPT)])
    return writer, judge


def fail(capsys, argv, code=2):
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == code
    return capsys.readouterr().err


def test_version(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert capsys.readouterr().out == VERSION


def test_help_russian(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--help"])
    assert info.value.code == 0
    output = capsys.readouterr().out
    assert output.startswith("использование: ruts-adapt [-h] -g КЛАСС [-n N] [--version] [ФАЙЛ]")
    assert "\nаргументы:\n" in output
    assert "\nпараметры:\n" in output
    assert "RUTS_ADAPT_WRITER_MODEL" in output
    assert "options" not in output
    assert "positional" not in output


def test_adapt_file(tmp_path, capsys, models):
    """Итоговый текст - в stdout, ход работы и таблица - в stderr"""
    source = tmp_path / "текст.txt"
    source.write_text(DECREE, encoding="utf-8")
    models(*fakes())
    main(["--grade", "2", str(source)])
    output = capsys.readouterr()
    assert output.out == f"{SIMPLE}\n"
    assert output.err.startswith("Исходный текст: «аспирантура (старше 22 лет)»")
    assert "Готово: попытка 1 на уровне цели, все факты сохранены" in output.err


def test_adapt_stdin(capsys, models, stdin):
    stdin(DECREE.encode())
    models(*fakes())
    main(["-g", "2", "-n", "1"])
    assert capsys.readouterr().out == f"{SIMPLE}\n"


def test_on_level(capsys, models, stdin):
    """Текст уже на уровне цели печатается как есть, модель не вызывается"""
    stdin(SIMPLE.encode())
    writer, judge = fakes()
    models(writer, judge)
    main(["-g", "2", "-"])
    output = capsys.readouterr()
    assert output.out == f"{SIMPLE}\n"
    assert output.err.endswith("Текст уже на уровне цели, переписывать не нужно\n")
    assert writer.requests == judge.requests == []


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ([], "нужны аргументы: -g/--grade"),
        (["-g"], "аргумент -g/--grade: нужно значение"),
        (["-g", "пятый"], "аргумент -g/--grade: нужен класс от 1 до 17, получено 'пятый'"),
        (["-g", "18"], "аргумент -g/--grade: нужен класс от 1 до 17, получено '18'"),
        (
            ["-g", "5", "-n", "0"],
            "аргумент -n/--attempts: нужно целое число от 1 до 10, получено '0'",
        ),
        (
            ["-g", "5", "-n", "x"],
            "аргумент -n/--attempts: нужно целое число от 1 до 10, получено 'x'",
        ),
        (["-g", "5", "a.txt", "b.txt"], "неизвестные аргументы: b.txt"),
        (["--version=1"], "аргумент --version: лишнее значение '1'"),
    ],
    ids=[
        "no-grade",
        "no-value",
        "grade-word",
        "grade-range",
        "attempts-zero",
        "attempts-word",
        "unrecognized",
        "explicit",
    ],
)
def test_argument_errors(capsys, argv, message):
    error = fail(capsys, argv)
    assert error.startswith("использование: ruts-adapt")
    assert error.endswith(f"\nruts-adapt: ошибка: {message}\n")


def test_error_untranslated(capsys):
    """Сообщение без перевода печатается как есть"""
    with pytest.raises(SystemExit) as info:
        ArgumentParser(prog="ruts-adapt").error("неверная цель")
    assert info.value.code == 2
    assert capsys.readouterr().err.endswith("\nruts-adapt: ошибка: неверная цель\n")


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        (lambda path: path / "нет.txt", "файл не найден"),
        (lambda path: path, "это каталог"),
        (lambda path: path / "cp1251.txt", "текст не в кодировке UTF-8"),
    ],
    ids=["missing", "directory", "encoding"],
)
def test_file_errors(tmp_path, capsys, make, reason):
    (tmp_path / "cp1251.txt").write_bytes(DECREE.encode("cp1251"))
    path = make(tmp_path)
    error = fail(capsys, ["-g", "2", str(path)])
    assert error.endswith(f"ruts-adapt: ошибка: не удалось прочитать {path}: {reason}\n")


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (PermissionError(13, "Permission denied"), "нет доступа"),
        (OSError(5, "Input/output error"), "[Errno 5] Input/output error"),
    ],
    ids=["permission", "other"],
)
def test_file_os_errors(capsys, monkeypatch, failure, reason):
    """Нет доступа - по-русски, прочие ошибки чтения - с сообщением системы"""

    def broken(encoding):
        raise failure

    path = SimpleNamespace(is_dir=lambda: False, read_text=broken)
    monkeypatch.setattr(cli, "Path", lambda file: path)
    error = fail(capsys, ["-g", "2", "текст.txt"])
    assert error.endswith(f"ruts-adapt: ошибка: не удалось прочитать текст.txt: {reason}\n")


@pytest.mark.parametrize(
    ("data", "tty", "message"),
    [
        (b"", True, "нет текста: укажите ФАЙЛ или передайте текст через stdin"),
        (DECREE.encode("cp1251"), False, "текст в stdin не в кодировке UTF-8"),
        (b"   \n", False, "В тексте нет букв: адаптировать нечего"),
        (b"Mama washed the frame", False, "Букв русского алфавита - только 0%"),
    ],
    ids=["tty", "encoding", "empty", "english"],
)
def test_input_errors(capsys, stdin, data, tty, message):
    stdin(data, tty)
    assert f"ruts-adapt: ошибка: {message}" in fail(capsys, ["-g", "2"])


def test_settings_error(capsys, monkeypatch, stdin):
    stdin(DECREE.encode())
    monkeypatch.setenv("RUTS_ADAPT_WRITER_EFFORT", "turbo")
    error = fail(capsys, ["-g", "2"])
    assert "ruts-adapt: ошибка: RUTS_ADAPT_WRITER_EFFORT для anthropic должна быть" in error


def test_model_profile_error(capsys, monkeypatch, stdin):
    """Усилие, которое модель не поддерживает, - ошибка настроек"""
    stdin(DECREE.encode())
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("RUTS_ADAPT_JUDGE_MODEL", "openai:gpt-4.1")
    monkeypatch.setenv("RUTS_ADAPT_JUDGE_EFFORT", "high")
    error = fail(capsys, ["-g", "2"])
    assert error.endswith(
        "ruts-adapt: ошибка: Модель openai:gpt-4.1 не поддерживает усилие 'high': "
        "оставьте усилие пустым\n"
    )


def test_model_create_error(capsys, monkeypatch, stdin):
    """Модель не создается, например без ключа API: код 1 и сообщение провайдера"""
    stdin(DECREE.encode())

    def missing_key(settings):
        raise RuntimeError("Missing credentials")

    monkeypatch.setattr(cli, "chat_model", missing_key)
    error = fail(capsys, ["-g", "2"], code=1)
    assert error == "ruts-adapt: не удалось создать модель: Missing credentials\n"


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (
            anthropic(stop_reason="refusal", category="bio"),
            "ошибка модели: Модель claude-opus-5-5 отказалась отвечать (причина: bio)",
        ),
        (
            ConnectionError("connection refused"),
            "ошибка модели: Модель fake не ответила: ConnectionError: connection refused",
        ),
    ],
    ids=["model", "provider"],
)
def test_run_errors(capsys, models, stdin, failure, message):
    stdin(DECREE.encode())
    writer, judge = fakes()
    judge.replies[0] = failure
    models(writer, judge)
    error = fail(capsys, ["-g", "2"], code=1)
    assert error.endswith(f"ruts-adapt: {message}\n")
    assert capsys.readouterr().out == ""


def test_program_error(models, stdin, monkeypatch):
    """Ошибка программы не выдается за ошибку модели: трейсбек остается"""
    stdin(DECREE.encode())
    models(*fakes())

    def broken(text):
        raise RuntimeError("ошибка в программе")

    monkeypatch.setattr("ruts_adapt.graph.measure", broken)
    with pytest.raises(RuntimeError, match="ошибка в программе"):
        main(["-g", "2"])


def test_tokens_with_retries(capsys, models, stdin, monkeypatch):
    """Итог считает токены всех вызовов модели, и повторенных тоже"""
    stdin(DECREE.encode())
    incomplete = anthropic(Verdict(checks=KEPT.checks[:1]))
    writer = FakeChatModel(replies=[anthropic(Draft(text=SIMPLE))])
    judge = FakeChatModel(replies=[anthropic(Facts(facts=FACTS)), incomplete, anthropic(KEPT)])
    models(writer, judge)
    monkeypatch.setattr(cli, "build_graph", lambda: build_graph(FAST))
    main(["-g", "2"])
    assert "Токены всех вызовов модели: 660\n" in capsys.readouterr().err


def test_interrupt_handler(models, stdin, monkeypatch):
    """На время работы графа Ctrl+C завершает процесс сразу, после - прежний обработчик"""
    stdin(DECREE.encode())
    seen = []

    def facts():
        seen.append(signal.getsignal(signal.SIGINT))
        return anthropic(Facts(facts=FACTS))

    writer, judge = fakes()
    judge.replies[0] = facts
    models(writer, judge)
    before = signal.getsignal(signal.SIGINT)
    main(["-g", "2"])
    assert seen == [cli._interrupt]
    assert signal.getsignal(signal.SIGINT) is before
    exits = []
    monkeypatch.setattr(cli.os, "_exit", exits.append)
    cli._interrupt(signal.SIGINT, None)
    assert exits == [130]


def test_interrupted(capsys, models, stdin):
    """Ctrl+C завершает команду с кодом 130 без трейсбека"""
    stdin(DECREE.encode())
    writer, judge = fakes()
    judge.replies[0] = KeyboardInterrupt()
    models(writer, judge)
    assert fail(capsys, ["-g", "2"], code=130).startswith("Исходный текст")


def test_interrupted_reading(capsys, monkeypatch):
    """Ctrl+C до графа - пока команда читает текст - тоже код 130 без трейсбека"""

    def interrupt(file):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_read", interrupt)
    assert fail(capsys, ["-g", "2"], code=130) == ""


def test_module(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ruts_adapt", "--version"])
    monkeypatch.delitem(sys.modules, "ruts_adapt.__main__", raising=False)
    with pytest.raises(SystemExit) as info:
        runpy.run_module("ruts_adapt", run_name="__main__")
    assert info.value.code == 0
    assert capsys.readouterr().out == VERSION


def test_module_import(monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "ruts_adapt.__main__", raising=False)
    importlib.import_module("ruts_adapt.__main__")
    assert capsys.readouterr().out == ""


class Console(io.TextIOWrapper):
    """Консоль Windows с однобайтной кодировкой"""

    def isatty(self) -> bool:
        return True


def test_output_errors_replace(monkeypatch):
    """Справка печатается и на консоли без кириллицы: символы заменяются, а не роняют команду"""
    monkeypatch.setenv("NO_COLOR", "1")
    buffer = io.BytesIO()
    stdout = Console(buffer, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    with pytest.raises(SystemExit):
        main(["--help"])
    stdout.flush()
    assert buffer.getvalue().startswith(b"?????????????: ruts-adapt")


def test_output_redirect_utf8(monkeypatch, models, stdin):
    """Итоговый текст в файл пишется в UTF-8, даже если кодировка системы однобайтная"""
    buffer = io.BytesIO()
    stdout = io.TextIOWrapper(buffer, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    stdin(DECREE.encode())
    models(*fakes())
    main(["-g", "2"])
    stdout.flush()
    assert buffer.getvalue().decode("utf-8") == f"{SIMPLE}{os.linesep}"
