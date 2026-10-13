"""
Живые прогоны: тексты набора, модели из окружения, прогон графа и стенда судьи

Живые тесты стоят денег и запускаются только make test-llm: тесты идут лишь
с маркером llm и переменной RUTS_ADAPT_LIVE=1, которую задает эта цель, -
любой другой запуск pytest их исключает или пропускает. Модели и провайдеры -
из переменных RUTS_ADAPT_*, как у команды. Тексты - из наборов ruTS
(скачиваются при первом запуске) и tests/data. Прогоны описаны здесь, а не
в тестах, чтобы проверить их на поддельных моделях до платного запуска

Стенд судьи: писатель пересказывает текст для класса читателя случая и по
заданию убирает и искажает факты с номерами из damage_plan, судья проверяет
пересказ, как в графе. Совпадение вердиктов с порчей - полнота и ложные
тревоги судьи
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import pytest
from langchain_core.callbacks import UsageMetadataCallbackHandler, get_usage_metadata_callback
from pydantic import BaseModel, Field
from ruts.datasets import SovChLit, TextsByGrade

from ruts_adapt import tasks
from ruts_adapt.graph import RETRY, Context, build_graph
from ruts_adapt.measure import Target, measure
from ruts_adapt.model import ModelError, ask, chat_model
from ruts_adapt.settings import Settings
from ruts_adapt.tasks import FactCheck, FactStatus
from tests.live_report import JudgeRun, Run

T = TypeVar("T")

DATA = Path(__file__).parent / "data"
KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
}
MIN_WORDS = 150
# Сколько фактов портит писатель на стенде судьи
DAMAGED = 5
DAMAGE_SYSTEM = """\
Ты готовишь контрольные пересказы для проверки судьи: судья сверяет пересказ со списком фактов \
исходного текста и должен заметить потерянные и искаженные факты.

Перескажи исходный текст для читателя заданного уровня связным русским языком того же жанра \
и сделай с фактами из задания ровно то, что там сказано:
- убрать - в пересказе нет ни этого факта, ни намека на него;
- исказить - в факте изменена одна деталь: участник, число, место, время или причина, так что \
утверждение стало ложным, но осталось правдоподобным.
Все остальные факты сохрани точно. В поле changes для каждого убранного и искаженного факта \
одной строкой напиши его номер и что с ним сделано."""
ACTIONS = {FactStatus.LOST: "Убрать", FactStatus.DISTORTED: "Исказить"}


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


class Damaged(BaseModel):
    """Пересказ с намеренно убранными и искаженными фактами"""

    text: str = Field(description="Пересказ целиком")
    changes: list[str] = Field(description="Номер и порча каждого убранного и искаженного факта")


def live_context() -> Context:
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


def run_graph(case: Case, context: Context, runs: list[Run]) -> dict[str, Any]:
    """Прогон графа на тексте случая; запись в runs - и тогда, когда граф упал"""
    text = case.load()
    target = Target.from_grade(case.grade)
    initial: dict[str, Any] = {"source": text, "target": target, "max_attempts": 3}
    state = initial
    critiques: list[str] = []
    error = None
    with get_usage_metadata_callback() as usage:
        try:
            for values in build_graph().stream(initial, context=context, stream_mode="values"):
                state = values
                # Критика остается в состоянии и после следующих шагов
                if values.get("critique") and values["critique"] not in critiques[-1:]:
                    critiques.append(values["critique"])
        except BaseException as failure:
            error = _error(failure)
            raise
        finally:
            runs.append(
                Run(
                    case=case.name,
                    genre=case.genre,
                    target=target,
                    source=text,
                    source_measurement=state.get("source_measurement") or measure(text),
                    facts=state.get("facts", []),
                    attempts=state.get("attempts", []),
                    critiques=critiques,
                    result=state.get("result"),
                    tokens=_tokens(usage),
                    error=error,
                )
            )
    return state


def run_bench(case: Case, context: Context, runs: list[JudgeRun]) -> JudgeRun:
    """Стенд судьи на тексте случая; запись в runs - и тогда, когда стенд упал"""
    text = case.load()
    facts: list[str] = []
    plan: dict[int, FactStatus] = {}
    damaged: Damaged | None = None
    checks: list[FactCheck] | None = None
    error = None
    with get_usage_metadata_callback() as usage:
        try:
            facts = retried(lambda: tasks.extract_facts(context.judge, text)).output
            plan = damage_plan(len(facts))
            prompt = damage_prompt(text, facts, Target.from_grade(case.grade), plan)
            damaged = retried(lambda: ask(context.writer, DAMAGE_SYSTEM, prompt, Damaged)).output
            draft = damaged.text
            checks = retried(lambda: tasks.judge(context.judge, facts, draft)).output
        except BaseException as failure:
            error = _error(failure)
            raise
        finally:
            run = JudgeRun(
                case=case.name,
                facts=facts,
                plan=plan,
                text=damaged.text if damaged else None,
                changes=damaged.changes if damaged else [],
                checks=checks,
                tokens=_tokens(usage),
                error=error,
            )
            runs.append(run)
    return run


def damage_plan(n_facts: int) -> dict[int, FactStatus]:
    """
    Факты, которые писатель портит на стенде судьи

    Описание:
        DAMAGED номеров равномерно по тексту: первый, третий и пятый из них
        убираются, второй и четвертый искажаются. В коротком списке номера
        совпадают, и испорченных фактов меньше

    Аргументы:
        n_facts (int): Число фактов

    Вывод:
        dict[int, FactStatus]: Номер факта с 1 и порча: LOST или DISTORTED

    Пример использования:
        >>> {number: str(damage) for number, damage in damage_plan(12).items()}
        {2: 'потерян', 4: 'искажен', 6: 'потерян', 8: 'искажен', 10: 'потерян'}
    """
    numbers = sorted({max(1, round(n_facts * k / (DAMAGED + 1))) for k in range(1, DAMAGED + 1)})
    return {
        number: FactStatus.LOST if index % 2 == 0 else FactStatus.DISTORTED
        for index, number in enumerate(numbers)
    }


def damage_prompt(text: str, facts: list[str], target: Target, plan: dict[int, FactStatus]) -> str:
    """Задание писателю стенда: уровень пересказа, исходник, факты и номера для порчи"""
    numbered = "\n".join(f"{number}. {fact}" for number, fact in enumerate(facts, 1))
    task = "\n".join(
        f"{action}: {', '.join(str(number) for number, kind in plan.items() if kind is damage)}"
        for damage, action in ACTIONS.items()
    )
    return (
        f"Перескажи исходный текст для читателя уровня «{target.level}».\n\n"
        f"<исходный_текст>\n{text}\n</исходный_текст>\n\n"
        f"<факты>\n{numbered}\n</факты>\n\n"
        f"<задание>\n{task}\n</задание>"
    )


def retried(call: Callable[[], T]) -> T:
    """Вызов с повтором, как у узлов графа: повторяется ModelError с retryable"""
    for _ in range(RETRY.max_attempts - 1):
        try:
            return call()
        except ModelError as error:
            if not error.retryable:
                raise
    return call()


def _error(failure: BaseException) -> str:
    return f"{type(failure).__name__}: {failure}".removesuffix(": ")


def _tokens(usage: UsageMetadataCallbackHandler) -> int:
    """Входные и выходные токены всех вызовов моделей за прогон"""
    return sum(
        item["input_tokens"] + item["output_tokens"] for item in usage.usage_metadata.values()
    )
