import importlib
import io
import os
import re
import runpy
import signal
import sqlite3
import sys
from importlib.metadata import version
from types import SimpleNamespace

import pytest

import ruts_adapt
from ruts_adapt import cli
from ruts_adapt.checkpoint import DATABASE, thread_lock
from ruts_adapt.cli import ArgumentParser, main
from ruts_adapt.graph import build_graph
from ruts_adapt.tasks import Draft, Facts, Verdict
from tests.fakes import FakeChatModel, anthropic
from tests.samples import DECREE, FACTS, KEPT, MEDIUM, SIMPLE
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


def run(argv, code=0):
    """Команда до конца: 0 - на уровне цели, NOT_ACCEPTED - лучшая непринятая, PAUSED - пауза"""
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == code


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
    assert output.startswith(
        "использование: ruts-adapt [-h] -g КЛАСС [-n N] [-r] [-t ПОТОК] [ФАЙЛ]\n"
        "       ruts-adapt -t ПОТОК [--approve | --reject ЗАМЕЧАНИЕ]"
    )
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
    run(["--grade", "2", str(source)])
    output = capsys.readouterr()
    assert output.out == f"{SIMPLE}\n"
    first, second = output.err.splitlines()[:2]
    assert re.fullmatch(
        r"Поток ([0-9a-f]{8}): продолжить после остановки - ruts-adapt --thread \1", first
    )
    assert second.startswith("Исходный текст: «аспирантура (старше 22 лет)»")
    assert "Готово: попытка 1 на уровне цели, все факты сохранены" in output.err


def test_adapt_stdin(capsys, models, stdin):
    stdin(DECREE.encode())
    models(*fakes())
    run(["-g", "2", "-n", "1"])
    assert capsys.readouterr().out == f"{SIMPLE}\n"


def test_on_level(capsys, models, stdin):
    """Текст уже на уровне цели печатается как есть, модель не вызывается"""
    stdin(SIMPLE.encode())
    writer, judge = fakes()
    models(writer, judge)
    run(["-g", "2", "-"])
    output = capsys.readouterr()
    assert output.out == f"{SIMPLE}\n"
    assert output.err.endswith("Текст уже на уровне цели, переписывать не нужно\n")
    assert writer.requests == judge.requests == []


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ([], "нужен аргумент -g/--grade"),
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
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("RUTS_ADAPT_JUDGE_MODEL", "openai:gpt-4.1")
    monkeypatch.setenv("RUTS_ADAPT_JUDGE_EFFORT", "high")
    error = fail(capsys, ["-g", "2"])
    assert error.endswith(
        "ruts-adapt: ошибка: Модель openai:gpt-4.1 не поддерживает усилие 'high': "
        "оставьте усилие пустым\n"
    )


def test_model_create_error(capsys, monkeypatch, stdin, data_dir):
    """Модель не создается: код 1 до первого шага, без подсказки продолжить несохраненный поток"""
    stdin(DECREE.encode())

    def broken(settings):
        raise RuntimeError("Missing credentials")

    monkeypatch.setattr(cli, "chat_model", broken)
    error = fail(capsys, ["-g", "2", "-t", "new"], code=1)
    assert error == "ruts-adapt: не удалось создать модель: Missing credentials\n"
    assert "поток new не найден" in fail(capsys, ["-t", "new", "--approve"])


def test_missing_key(capsys, monkeypatch, stdin):
    """Без ключа API провайдера - ошибка настроек сразу, а не на первом вызове модели"""
    stdin(DECREE.encode())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    error = fail(capsys, ["-g", "2"])
    assert error.endswith(
        "ruts-adapt: ошибка: Нет ключа API для модели anthropic:claude-opus-5-5: задайте "
        "переменную окружения ANTHROPIC_API_KEY\n"
    )
    assert "Поток" not in error


def test_model_validation_error(capsys, monkeypatch, stdin):
    """Проверка параметров модели pydantic - без английской сводки pydantic"""
    stdin(DECREE.encode())

    def invalid(settings):
        from pydantic import BaseModel

        class Options(BaseModel):
            timeout: int

        Options(timeout="долго")

    monkeypatch.setattr(cli, "chat_model", invalid)
    error = fail(capsys, ["-g", "2"])
    assert error.endswith(
        "ruts-adapt: ошибка: не удалось создать модель: Input should be a valid integer, "
        "unable to parse string as an integer\n"
    )


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (
            anthropic(stop_reason="refusal", category="bio"),
            "ошибка модели: Модель claude-opus-5-5 отказалась отвечать (причина: bio)",
        ),
        (
            PermissionError("invalid x-api-key"),
            "ошибка модели: Модель fake не ответила: PermissionError: invalid x-api-key",
        ),
    ],
    ids=["model", "provider"],
)
def test_run_errors(capsys, models, stdin, failure, message):
    stdin(DECREE.encode())
    writer, judge = fakes()
    judge.replies[0] = failure
    models(writer, judge)
    error = fail(capsys, ["-g", "2", "-t", "x"], code=1)
    assert error.endswith(f"ruts-adapt: {message}. Продолжить: ruts-adapt --thread x\n")
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
    monkeypatch.setattr(cli, "build_graph", lambda **kwargs: build_graph(FAST, **kwargs))
    run(["-g", "2"])
    assert "Токены вызовов модели в этом запуске: 660\n" in capsys.readouterr().err


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
    run(["-g", "2"])
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
    assert "Исходный текст" in fail(capsys, ["-g", "2"], code=130)


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
    run(["-g", "2"])
    stdout.flush()
    assert buffer.getvalue().decode("utf-8") == f"{SIMPLE}{os.linesep}"


def test_review_flow(capsys, models, stdin):
    """--review: пауза перед итогом, затем решение по потоку и итог; завершенный поток - итог"""
    stdin(DECREE.encode())
    models(*fakes())
    run(["-g", "2", "-r", "-t", "demo"], code=cli.PAUSED)
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.splitlines()[-3:] == [
        "Токены вызовов модели в этом запуске: 495",
        "Принять: ruts-adapt --thread demo --approve",
        'Вернуть с замечанием: ruts-adapt --thread demo --reject "замечание"',
    ]
    models(FakeChatModel(replies=[]), FakeChatModel(replies=[]))
    run(["-t", "demo", "--approve"])
    output = capsys.readouterr()
    assert output.out == f"{SIMPLE}\n"
    assert output.err.startswith("Продолжаем поток demo\nПринято проверяющим\n")
    run(["-t", "demo"])
    output = capsys.readouterr()
    assert (output.out, output.err) == (f"{SIMPLE}\n", "Поток demo уже завершен\n")


def test_review_reject(capsys, models, stdin):
    stdin(DECREE.encode())
    models(*fakes())
    run(["-g", "2", "-r", "-t", "demo"], code=cli.PAUSED)
    capsys.readouterr()
    writer = FakeChatModel(replies=[anthropic(Draft(text=SIMPLE + " Это все."))])
    models(writer, FakeChatModel(replies=[anthropic(KEPT)]))
    run(["-t", "demo", "--reject", "короче"], code=cli.PAUSED)
    output = capsys.readouterr()
    assert output.out == ""
    assert "Возвращено с замечанием: короче" in output.err
    assert "Попытка 2 ждет решения" in output.err
    assert "Замечание проверяющего: короче" in writer.requests[0][1].content


def test_resume_after_error(capsys, models, stdin):
    """Ошибка модели подсказывает, как продолжить, и поток продолжается с места остановки"""
    stdin(DECREE.encode())
    writer, judge = fakes()
    judge.replies[1] = anthropic(stop_reason="refusal", category="bio")
    models(writer, judge)
    error = fail(capsys, ["-g", "2", "-t", "x"], code=1)
    assert error.endswith("Продолжить: ruts-adapt --thread x\n")
    judge = FakeChatModel(replies=[anthropic(KEPT)])
    models(FakeChatModel(replies=[]), judge)
    run(["-t", "x"])
    output = capsys.readouterr()
    assert output.out == f"{SIMPLE}\n"
    assert output.err.startswith("Продолжаем поток x\n")
    assert len(judge.requests) == 1


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["-t", "nope", "--approve"], "поток nope не найден в {data_dir}"),
        (["-t", "nope"], "поток nope не найден в {data_dir}; новый поток - с -g КЛАСС"),
        (["-t", "nope", "--reject", " "], "замечание --reject не должно быть пустым"),
        (
            ["-t", "nope", "--approve", "--reject", "x"],
            "аргумент --reject: нельзя вместе с --approve",
        ),
    ],
    ids=["unknown", "unknown-resume", "empty-reject", "both"],
)
def test_thread_errors(capsys, data_dir, argv, message):
    message = message.format(data_dir=data_dir)
    assert fail(capsys, argv).endswith(f"ruts-adapt: ошибка: {message}\n")


@pytest.fixture
def waiting(capsys, models, stdin):
    """Поток demo, который ждет решения"""
    stdin(DECREE.encode())
    models(*fakes())
    run(["-g", "2", "-r", "-t", "demo"], code=cli.PAUSED)
    capsys.readouterr()


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["-t", "demo", "-g", "5", "-n", "2", "-r", "текст.txt", "--approve"],
            "поток demo уже есть, продолжение не принимает: ФАЙЛ, --grade, --attempts, --review",
        ),
    ],
    ids=["arguments"],
)
def test_waiting_errors(waiting, capsys, argv, message):
    assert fail(capsys, argv).endswith(f"ruts-adapt: ошибка: {message}\n")


def test_waiting_shown(waiting, capsys, models):
    """Поток без решения показывает кандидата снова, модели не создаются"""
    models(FakeChatModel(replies=[]), FakeChatModel(replies=[]))
    run(["-t", "demo"], code=cli.PAUSED)
    output = capsys.readouterr()
    assert output.out == ""
    assert "Попытка 1 ждет решения - на уровне цели, все факты сохранены. Черновик:" in output.err
    assert output.err.endswith(
        'Вернуть с замечанием: ruts-adapt --thread demo --reject "замечание"\n'
    )
    assert "Токены вызовов модели в этом запуске: 0\n" in output.err


def test_decision_not_waiting(capsys, models, stdin):
    """Решение передается только потоку, который его ждет"""
    stdin(DECREE.encode())
    models(*fakes())
    run(["-g", "2", "-t", "done"])
    error = fail(capsys, ["-t", "done", "--approve"])
    assert error.endswith("ruts-adapt: ошибка: поток done уже завершен\n")
    stdin(DECREE.encode())
    writer, judge = fakes()
    judge.replies[1] = anthropic(stop_reason="refusal")
    models(writer, judge)
    fail(capsys, ["-g", "2", "-t", "broken"], code=1)
    error = fail(capsys, ["-t", "broken", "--reject", "короче"])
    assert error.endswith("ruts-adapt: ошибка: поток broken не ждет решения\n")


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["-g", "2", "-t", "статья 1"],
            "аргумент -t/--thread: имя потока - буквы, цифры, «-», «_» и «.», начиная с буквы "
            "или цифры, получено 'статья 1'",
        ),
        (["-g", "2", "--thread=-x"], "аргумент -t/--thread: имя потока - буквы, цифры"),
        (["--approve"], "решение передается сохраненному потоку: нужен --thread"),
    ],
    ids=["space", "dash", "no-thread"],
)
def test_thread_argument_errors(capsys, argv, message):
    assert f"ruts-adapt: ошибка: {message}" in fail(capsys, argv)


def test_no_database_on_errors(capsys, data_dir):
    """Без --thread ошибки аргументов и текста ловятся до открытия хранилища"""
    fail(capsys, [])
    assert not data_dir.exists()


@pytest.mark.parametrize("kind", ["file", "not-sqlite", "foreign-tables"])
def test_storage_errors(capsys, stdin, data_dir, kind):
    """Недоступное хранилище - понятная ошибка, а не трейсбек"""
    if kind == "file":
        data_dir.write_text("")
    elif kind == "not-sqlite":
        data_dir.mkdir()
        (data_dir / "checkpoints.sqlite").write_bytes(b"not a database")
    else:
        data_dir.mkdir()
        with sqlite3.connect(data_dir / "checkpoints.sqlite") as connection:
            connection.execute("CREATE TABLE checkpoints (name TEXT)")
    stdin(DECREE.encode())
    error = fail(capsys, ["-g", "2"])
    assert f"ruts-adapt: ошибка: не удалось открыть хранилище потоков в {data_dir}: " in error


def test_storage_write_error(capsys, models, stdin, monkeypatch, data_dir):
    """База открылась, но запись падает (только чтение, диск, блокировка) - ошибка без трейсбека"""
    stdin(DECREE.encode())
    models(*fakes())

    def readonly(*args):
        raise sqlite3.OperationalError("attempt to write a readonly database")

    monkeypatch.setattr(cli, "adapt", readonly)
    error = fail(capsys, ["-g", "2"], code=1)
    assert error.endswith(
        f"ruts-adapt: не удалось сохранить поток в {data_dir}: attempt to write a readonly database\n"
    )


def test_old_version(capsys, models, stdin, monkeypatch):
    """Поток, сохраненный другой версией формата, не продолжается: классы могли измениться"""
    stdin(DECREE.encode())
    models(*fakes())
    run(["-g", "2", "-r", "-t", "old"], code=cli.PAUSED)
    monkeypatch.setattr(cli, "STATE_VERSION", cli.STATE_VERSION + 1)
    error = fail(capsys, ["-t", "old", "--approve"])
    assert error.endswith(
        "ruts-adapt: ошибка: поток old сохранен другой версией ruts-adapt: начните новый поток "
        "с другим именем\n"
    )


def test_data_dir_error(capsys, monkeypatch):
    monkeypatch.setenv("RUTS_ADAPT_DATA_DIR", "data")
    error = fail(capsys, ["-g", "2"])
    assert "RUTS_ADAPT_DATA_DIR должна быть абсолютным путем, получено 'data'" in error


def test_threads_saved(capsys, models, stdin, data_dir):
    """Потоки хранятся в каталоге данных"""
    stdin(DECREE.encode())
    models(*fakes())
    run(["-g", "2"])
    assert (data_dir / "checkpoints.sqlite").is_file()


def test_best_exit_code(capsys, models, stdin):
    """Лучшая попытка не на уровне цели: текст в stdout, код NOT_ACCEPTED - и при повторном показе"""
    stdin(DECREE.encode())
    writer = FakeChatModel(replies=[anthropic(Draft(text=MEDIUM))])
    judge = FakeChatModel(replies=[anthropic(Facts(facts=FACTS)), anthropic(KEPT)])
    models(writer, judge)
    run(["-g", "2", "-n", "1", "-t", "best"], code=cli.NOT_ACCEPTED)
    assert capsys.readouterr().out == f"{MEDIUM}\n"
    run(["-t", "best"], code=cli.NOT_ACCEPTED)
    assert capsys.readouterr().out == f"{MEDIUM}\n"


@pytest.mark.parametrize("source", ["file", "stdin"])
def test_bom_crlf(tmp_path, capsys, models, stdin, source):
    """Файл и stdin читаются одинаково: без BOM и с переводами строк LF"""
    data = "\ufeffЕсть новые правила.\r\nФирмам дают деньги.\r\n".encode()
    models(*fakes())
    if source == "file":
        path = tmp_path / "текст.txt"
        path.write_bytes(data)
        run(["-g", "2", str(path)])
    else:
        stdin(data)
        run(["-g", "2"])
    assert capsys.readouterr().out == "Есть новые правила.\nФирмам дают деньги.\n"


def test_broken_pipe(tmp_path, monkeypatch):
    """Читатель канала закрылся: код 1 без трейсбека, остаток вывода - в никуда"""

    def closed(parser, args):
        raise BrokenPipeError

    monkeypatch.setattr(cli, "_run", closed)
    with (tmp_path / "out").open("w") as stdout, (tmp_path / "err").open("w") as stderr:
        monkeypatch.setattr(sys, "stdout", stdout)
        monkeypatch.setattr(sys, "stderr", stderr)
        with pytest.raises(SystemExit) as info:
            main(["-g", "2"])
    assert info.value.code == 1


def test_thread_busy(capsys, data_dir):
    """Поток, открытый другим процессом, не запускается второй раз"""
    with thread_lock(data_dir, "demo"):
        error = fail(capsys, ["-t", "demo"])
    assert error.endswith("ruts-adapt: ошибка: поток demo уже выполняется в другом процессе\n")


def test_corrupted_thread(capsys, models, stdin, data_dir):
    stdin(DECREE.encode())
    models(*fakes())
    run(["-g", "2", "-t", "x"])
    with sqlite3.connect(data_dir / DATABASE) as connection:
        connection.execute("UPDATE checkpoints SET checkpoint = X'C1C1C1' WHERE thread_id = 'x'")
    error = fail(capsys, ["-t", "x"])
    assert error.endswith(
        "ruts-adapt: ошибка: поток x поврежден: начните новый поток с другим именем\n"
    )


def test_stderr_redirect_utf8(monkeypatch, models, stdin):
    """Ход работы в файл - тоже в UTF-8: на паузе там и черновик"""
    buffer = io.BytesIO()
    stderr = io.TextIOWrapper(buffer, encoding="cp1252")
    monkeypatch.setattr(sys, "stderr", stderr)
    stdin(DECREE.encode())
    models(*fakes())
    run(["-g", "2", "-r"], code=cli.PAUSED)
    stderr.flush()
    assert "Попытка 1 ждет решения" in buffer.getvalue().decode("utf-8")


def test_pruned(models, stdin, data_dir):
    """После запуска у потока остается один чекпоинт, и поток по-прежнему продолжается"""
    stdin(DECREE.encode())
    models(*fakes())
    run(["-g", "2", "-r", "-t", "demo"], code=cli.PAUSED)
    with sqlite3.connect(data_dir / DATABASE) as connection:
        (count,) = connection.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = 'demo'"
        ).fetchone()
    assert count == 1
    run(["-t", "demo", "--approve"])
