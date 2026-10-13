import os
from pathlib import Path

import pytest

from tests.live_report import Run, report

LIVE_RUNS = pytest.StashKey[list[Run]]()


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    """Каталог данных теста вместо каталога пользователя: потоки адаптации пишутся сюда"""
    path = tmp_path / "data"
    monkeypatch.setenv("RUTS_ADAPT_DATA_DIR", str(path))
    return path


def pytest_configure(config):
    config.stash[LIVE_RUNS] = []


@pytest.fixture
def live_runs(request):
    """Прогоны живых тестов для отчета в конце сессии"""
    return request.config.stash[LIVE_RUNS]


def pytest_terminal_summary(terminalreporter, config):
    """Таблица живых прогонов; с RUTS_ADAPT_LIVE_REPORT - еще и в файл"""
    runs = config.stash[LIVE_RUNS]
    if not runs:
        return
    table = report(runs)
    terminalreporter.write_sep("=", "живые прогоны")
    terminalreporter.write_line(table)
    if path := os.environ.get("RUTS_ADAPT_LIVE_REPORT"):
        Path(path).write_text(table + "\n", encoding="utf-8")
