import importlib
import io
import runpy
import sys

import pytest

import ruts_adapt
from ruts_adapt.cli import ArgumentParser, main


def test_version(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert capsys.readouterr().out == f"ruts-adapt {ruts_adapt.__version__}\n"


def test_no_arguments(capsys):
    main([])
    assert capsys.readouterr().out.startswith("использование: ruts-adapt [-h] [--version]")


def test_help_russian(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--help"])
    assert info.value.code == 0
    output = capsys.readouterr().out
    assert output.startswith("использование: ruts-adapt [-h] [--version]")
    assert "LLM-судья" in output
    assert "\nпараметры:\n" in output
    assert "options" not in output


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["текст", "--x"], "неизвестные аргументы: текст --x"),
        (["--version=1"], "аргумент --version: лишнее значение '1'"),
    ],
    ids=["unrecognized", "explicit"],
)
def test_errors_russian(capsys, argv, message):
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == 2
    error = capsys.readouterr().err
    assert error.startswith("использование: ruts-adapt")
    assert error.endswith(f"\nruts-adapt: ошибка: {message}\n")


def test_error_untranslated(capsys):
    """Сообщение без перевода печатается как есть"""
    with pytest.raises(SystemExit) as info:
        ArgumentParser(prog="ruts-adapt").error("неверная цель")
    assert info.value.code == 2
    assert capsys.readouterr().err.endswith("\nruts-adapt: ошибка: неверная цель\n")


def test_module(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ruts_adapt", "--version"])
    monkeypatch.delitem(sys.modules, "ruts_adapt.__main__", raising=False)
    with pytest.raises(SystemExit) as info:
        runpy.run_module("ruts_adapt", run_name="__main__")
    assert info.value.code == 0
    assert capsys.readouterr().out == f"ruts-adapt {ruts_adapt.__version__}\n"


def test_module_import(monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "ruts_adapt.__main__", raising=False)
    importlib.import_module("ruts_adapt.__main__")
    assert capsys.readouterr().out == ""


def test_output_errors_replace(monkeypatch):
    """Справка печатается и на консоли без кириллицы: символы заменяются, а не роняют команду"""
    buffer = io.BytesIO()
    stdout = io.TextIOWrapper(buffer, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    with pytest.raises(SystemExit):
        main(["--help"])
    stdout.flush()
    assert buffer.getvalue().startswith(b"?????????????: ruts-adapt")
