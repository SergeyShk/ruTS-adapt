import os
from pathlib import Path

import pytest

from ruts_adapt.graph import Context
from tests.live import live_context
from tests.live_report import JudgeRun, Run, details, summary

LIVE_RUNS = pytest.StashKey[list[Run]]()
JUDGE_RUNS = pytest.StashKey[list[JudgeRun]]()


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    """Каталог данных теста вместо каталога пользователя: потоки адаптации пишутся сюда"""
    path = tmp_path / "data"
    monkeypatch.setenv("RUTS_ADAPT_DATA_DIR", str(path))
    return path


def pytest_configure(config):
    config.stash[LIVE_RUNS] = []
    config.stash[JUDGE_RUNS] = []


@pytest.fixture(scope="session")
def context() -> Context:
    """Модели живых прогонов; без RUTS_ADAPT_LIVE=1 или ключа провайдера - пропуск"""
    return live_context()


@pytest.fixture
def live_runs(request):
    """Прогоны графа для отчета в конце сессии"""
    return request.config.stash[LIVE_RUNS]


@pytest.fixture
def judge_runs(request):
    """Прогоны стенда судьи для отчета в конце сессии"""
    return request.config.stash[JUDGE_RUNS]


def pytest_terminal_summary(terminalreporter, config):
    """Таблицы живых прогонов; с RUTS_ADAPT_LIVE_REPORT - еще и файл с текстами"""
    runs, judge_runs = config.stash[LIVE_RUNS], config.stash[JUDGE_RUNS]
    if not runs and not judge_runs:
        return
    tables = summary(runs, judge_runs)
    terminalreporter.write_sep("=", "живые прогоны")
    terminalreporter.write_line(tables)
    if path := os.environ.get("RUTS_ADAPT_LIVE_REPORT"):
        text = f"{tables}\n\n{details(runs, judge_runs)}\n"
        Path(path).write_text(text, encoding="utf-8")
