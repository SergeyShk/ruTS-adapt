"""
Живые прогоны графа на настоящей модели: только вручную, с ключом, они стоят денег

Запуск - только make test-llm: тесты идут лишь с маркером llm и переменной
RUTS_ADAPT_LIVE=1, которую задает эта цель, - любой другой запуск pytest их
исключает или пропускает. Модели и провайдеры - из переменных
RUTS_ADAPT_*, как у команды. Тексты - из наборов ruTS (скачиваются при первом
запуске) и tests/data; в конце сессии pytest печатает таблицу прогонов, а с
RUTS_ADAPT_LIVE_REPORT записывает ее в этот файл
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import get_usage_metadata_callback
from ruts.datasets import SovChLit, TextsByGrade

from ruts_adapt.attempts import Outcome
from ruts_adapt.graph import Context, build_graph
from ruts_adapt.measure import Target, measure
from ruts_adapt.model import chat_model
from ruts_adapt.settings import Settings
from tests.live_report import Run

pytestmark = pytest.mark.llm

DATA = Path(__file__).parent / "data"
KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
}
MIN_WORDS = 150


def excerpt(text: str) -> str:
    """Абзацы с начала текста, пока в них не наберется MIN_WORDS слов"""
    paragraphs: list[str] = []
    for paragraph in text.split("\n"):
        paragraphs.append(paragraph)
        if len(" ".join(paragraphs).split()) >= MIN_WORDS:
            break
    return "\n".join(paragraphs)


def texts_by_grade(subject: str) -> Callable[[], str]:
    def load() -> str:
        dataset = TextsByGrade()
        dataset.download()
        (record,) = [item for item in dataset.get_records() if item["subject"] == subject]
        return excerpt(record["text"])

    return load


def sov_chrest_lit(subject: str) -> Callable[[], str]:
    def load() -> str:
        dataset = SovChLit()
        dataset.download()
        (record,) = [item for item in dataset.get_records() if item["subject"] == subject]
        return excerpt(record["text"])

    return load


@dataclass(frozen=True)
class Case:
    """Текст, его жанр, источник и класс читателя; slug - имя теста"""

    slug: str
    name: str
    genre: str
    load: Callable[[], str]
    grade: int


CASES = [
    Case("fox", "Лиса и журавль", "сказка", texts_by_grade("Лиса и журавль"), 10),
    Case("filippok", "Толстой. Филиппок", "рассказ для 1 класса", sov_chrest_lit("Филиппок I"), 7),
    Case("olesya", "Куприн. Олеся", "классическая проза", texts_by_grade("Куприн. Олеся"), 4),
    Case(
        "sunstroke",
        "Бунин. Солнечный удар",
        "классическая проза",
        texts_by_grade("Бунин. Солнечный удар"),
        2,
    ),
    Case("dorset", "Дорсет", "энциклопедия", texts_by_grade("Дорсет"), 5),
    Case(
        "antarctica",
        "Деятельность в Антарктике",
        "официальный документ",
        texts_by_grade("О разрешениях на осуществление деятельности в Антарктике"),
        8,
    ),
    Case(
        "microsoft",
        "Условия Microsoft",
        "правовые условия",
        texts_by_grade("Условия использования на сайте Microsoft"),
        10,
    ),
    Case(
        "photosynthesis",
        "Фотосинтез",
        "научно-популярный",
        lambda: (DATA / "photosynthesis.txt").read_text(encoding="utf-8").strip(),
        2,
    ),
]


@pytest.fixture(scope="module")
def context() -> Context:
    """Модели из переменных RUTS_ADAPT_*; без RUTS_ADAPT_LIVE=1 или ключа провайдера - пропуск"""
    # Второй замок после маркера: чужое выражение -m заменило бы -m 'not llm' из addopts
    if os.environ.get("RUTS_ADAPT_LIVE") != "1":
        pytest.skip("живые прогоны тратят деньги и запускаются только make test-llm")
    settings = Settings.from_env()
    for role in (settings.writer, settings.judge):
        key = KEYS.get(role.provider)
        if key and not os.environ.get(key):
            pytest.skip(f"нет {key} для {role.model}")
    return Context(writer=chat_model(settings.writer), judge=chat_model(settings.judge))


@pytest.mark.parametrize("case", CASES, ids=[case.slug for case in CASES])
def test_live(case, context, live_runs):
    """Граф доходит до итога, и итог ближе к цели, чем исходник"""
    text = case.load()
    target = Target.from_grade(case.grade)
    initial: dict[str, Any] = {"source": text, "target": target, "max_attempts": 3}
    state = initial
    error = None
    with get_usage_metadata_callback() as usage:
        # Поток состояний: если граф упадет, в отчет попадут сделанные попытки и токены
        try:
            for values in build_graph().stream(initial, context=context, stream_mode="values"):
                state = values
        except BaseException as failure:
            error = f"{type(failure).__name__}: {failure}".removesuffix(": ")
            raise
        finally:
            tokens = sum(
                item["input_tokens"] + item["output_tokens"]
                for item in usage.usage_metadata.values()
            )
            live_runs.append(
                Run(
                    case.name,
                    case.genre,
                    target,
                    text,
                    state.get("source_measurement") or measure(text),
                    len(state.get("facts", [])),
                    state.get("attempts", []),
                    state.get("result"),
                    tokens,
                    error,
                )
            )
    result = state["result"]
    source_gap = target.gap(state["source_measurement"].grade)
    if result.outcome is Outcome.ON_LEVEL:
        assert source_gap == 0
        return
    assert state["facts"]
    assert result.attempt is not None
    assert abs(result.attempt.gap) < abs(source_gap)
    if result.outcome is Outcome.ACCEPTED:
        assert result.attempt.accepted
