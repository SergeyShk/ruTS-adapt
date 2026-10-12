import math
from dataclasses import replace

import pytest

from ruts_adapt.attempts import Attempt, best, critique, mtld_drop
from ruts_adapt.measure import Target, measure
from ruts_adapt.tasks import FactCheck, FactStatus
from tests.samples import DECREE, FACTS, KEPT, MEDIUM, SIMPLE, verdict

TARGET = Target.from_grade(2)


def attempt(number=1, text=SIMPLE, checks=KEPT.checks, gap=None):
    measurement = measure(text)
    gap = TARGET.gap(measurement.grade) if gap is None else gap
    return Attempt(number, text, measurement, checks, gap, 330)


def test_accepted():
    assert attempt().accepted
    assert attempt().lost == attempt().distorted == []


@pytest.mark.parametrize(
    ("text", "checks"),
    [
        (MEDIUM, KEPT.checks),
        (SIMPLE, verdict(FactStatus.KEPT, FactStatus.LOST).checks),
        (SIMPLE, verdict(FactStatus.DISTORTED, FactStatus.KEPT).checks),
    ],
    ids=["level", "lost", "distorted"],
)
def test_not_accepted(text, checks):
    """Принимается только попытка на уровне цели без искаженных и потерянных фактов"""
    assert not attempt(text=text, checks=checks).accepted


def test_lost_distorted():
    checks = verdict(FactStatus.DISTORTED, FactStatus.LOST).checks
    assert attempt(checks=checks).distorted == [checks[0]]
    assert attempt(checks=checks).lost == [checks[1]]


def test_best():
    """Сначала сохранность смысла, затем близость к цели, затем более ранняя попытка"""
    lossless_far = attempt(1, gap=5.0)
    lossy_close = attempt(2, checks=verdict(FactStatus.KEPT, FactStatus.LOST).checks, gap=0.0)
    lossless_near = attempt(3, gap=-1.5)
    lossless_near_late = attempt(4, gap=1.5)
    assert best([lossless_far, lossy_close]) is lossless_far
    assert best([lossless_far, lossy_close, lossless_near]) is lossless_near
    assert best([lossless_near_late, lossless_near]) is lossless_near
    assert best([lossy_close]) is lossy_close


def test_critique_too_complex():
    source = measure(DECREE)
    text = critique(attempt(text=MEDIUM, checks=KEPT.checks), source, TARGET, FACTS)
    assert text.splitlines() == [
        "Попытка 1 читается как «аспирантура (старше 22 лет)» (сводный класс 25.5), "
        "цель - «1-3-й класс (6-8 лет)»: текст все еще сложнее цели на 22.5 класса.",
        "Сейчас в среднем 13.0 слова в предложении (в исходном 18.0), 4.00 слога и 9.08 буквы "
        "в слове (3.61 и 8.33), 46% слов от пяти слогов (50%).",
        "Делите длинные предложения и заменяйте длинные и многосложные слова короткими.",
    ]


def test_critique_too_simple():
    target = Target.from_grade(10)
    current = attempt(gap=target.gap(measure(SIMPLE).grade))
    text = critique(current, measure(DECREE), target, FACTS)
    assert "текст получился проще цели на 9.0 класса" in text
    assert text.endswith(
        "Объединяйте короткие предложения и используйте более точные и длинные слова."
    )


def test_critique_facts():
    """На уровне цели - только факты: искаженные с цитатой и потерянные"""
    checks = [
        FactCheck(fact=1, status=FactStatus.DISTORTED, quote="правила о налогах"),
        FactCheck(fact=2, status=FactStatus.LOST, quote=""),
    ]
    text = critique(attempt(2, checks=checks), measure(DECREE), TARGET, FACTS)
    assert text.splitlines() == [
        "Попытка 2 читается как «1-3-й класс (6-8 лет)» (сводный класс 1.0), "
        "цель - «1-3-й класс (6-8 лет)»: уровень достигнут, сохраните его.",
        "Искажены факты, верните их точный смысл:",
        "1. Постановление устанавливает порядок субсидий В черновике: «правила о налогах»",
        "Потеряны факты, верните их:",
        "2. Субсидии возмещают недополученные доходы",
    ]


LONG = measure("Кот спит на окне, а пес лежит у двери. " * 20)


def test_mtld_drop():
    drop = mtld_drop(LONG, measure("Кот спит. Кот спит. " * 50))
    assert drop == pytest.approx(1 - 10.0 / LONG.mtld)


@pytest.mark.parametrize(
    "result",
    [
        measure("Кот спит. " * 5),
        replace(LONG, mtld=math.inf),
        replace(LONG, mtld=LONG.mtld * 2),
        LONG,
    ],
    ids=["short", "inf", "rise", "same"],
)
def test_mtld_no_drop(result):
    """Короткий текст, неопределенный MTLD и рост разнообразия - не падение"""
    assert mtld_drop(LONG, result) is None


def test_mtld_zero_source():
    assert mtld_drop(replace(LONG, mtld=0.0), LONG) is None
