import math
from dataclasses import replace

import pytest

from ruts_adapt.attempts import Attempt, Outcome, Result, best, candidate, critique, mtld_drop
from ruts_adapt.measure import Target, measure
from ruts_adapt.tasks import FactCheck, FactStatus
from tests.samples import DECREE, FACTS, KEPT, MEDIUM, SIMPLE, verdict

TARGET = Target.from_grade(2)
SIMPLE_M = measure(SIMPLE)
TERMS = "Хлоропласты синтезируют углеводы. Фотосинтез преобразует электромагнитную энергию."


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


def test_candidate():
    """Итог - последняя попытка, если принята, иначе лучшая"""
    lossless_far = attempt(1, gap=5.0)
    accepted = attempt(2)
    assert candidate([lossless_far, accepted]) == Result(Outcome.ACCEPTED, SIMPLE, accepted)
    lossy = attempt(2, checks=verdict(FactStatus.KEPT, FactStatus.LOST).checks)
    assert candidate([lossless_far, lossy]) == Result(Outcome.BEST, SIMPLE, lossless_far)


def test_candidate_skips_rejected():
    """Возвращенная проверяющим попытка не становится кандидатом снова"""
    accepted = attempt(1)
    worse = attempt(2, text=MEDIUM)
    assert candidate([accepted, worse], rejected=[1]) == Result(Outcome.BEST, MEDIUM, worse)
    assert candidate([accepted, worse]).attempt is accepted


def test_critique_too_complex():
    current = attempt(text=MEDIUM, checks=KEPT.checks)
    text = critique(current, [current], measure(DECREE), TARGET, FACTS)
    assert text.splitlines() == [
        "Попытка 1 читается как «аспирантура (старше 22 лет)» (сводный класс 25.5), "
        "цель - «1-3-й класс (6-8 лет)»: текст сложнее цели на 22.5 класса.",
        "Сейчас в среднем 13.0 слова в предложении (в исходном 18.0), 4.00 слога и 9.08 буквы "
        "в слове (3.61 и 8.33), 46% слов от пяти слогов (50%).",
        "В текстах для этого уровня обычно 7-10 слов в предложении, 4.7 буквы в слове "
        "и не больше 2% слов от пяти слогов.",
        "Делите длинные предложения и заменяйте длинные и многосложные слова короткими.",
    ]


def test_critique_short_sentences_complex():
    """Сложный текст с короткими предложениями: упрощать слова, а не дробить фразы"""
    current = attempt(text=TERMS)
    assert current.gap > 0
    assert current.measurement.words_per_sent <= TARGET.norm.words_per_sent[1]
    text = critique(current, [current], measure(DECREE), TARGET, FACTS)
    assert text.splitlines()[-1] == (
        "Предложения уже не длиннее, чем в текстах этого уровня: не дробите их дальше, "
        "а заменяйте длинные слова и термины короткими и простыми."
    )


def test_critique_overshoot():
    """Текст усложняли и перескочили цель: просто сложнее цели, а не «все еще»"""
    target = Target.from_grade(5)
    current = attempt(text=MEDIUM, gap=target.gap(measure(MEDIUM).grade))
    text = critique(current, [current], measure(SIMPLE), target, FACTS)
    assert "текст сложнее цели на 19.5 класса." in text
    assert "все еще" not in text


def test_critique_too_simple():
    target = Target.from_grade(10)
    current = attempt(gap=target.gap(measure(SIMPLE).grade))
    text = critique(current, [current], measure(DECREE), target, FACTS)
    assert "текст проще цели на 9.0 класса" in text
    assert "В текстах для этого уровня" not in text
    assert text.endswith(
        "Объединяйте короткие предложения и используйте более точные и длинные слова."
    )


def test_critique_short_sentences_simple():
    """Простой текст короче норм ступени: объединять предложения, слова оставить простыми"""
    target = Target.from_grade(5)
    current = attempt(gap=target.gap(measure(SIMPLE).grade))
    text = critique(current, [current], measure(DECREE), target, FACTS)
    assert text.splitlines()[-1] == (
        "Предложения короче, чем в текстах этого уровня: объединяйте их, "
        "а слова оставляйте простыми."
    )


def test_critique_between():
    """Попытки по обе стороны ступени: признаки между ними и история всех попыток"""
    target = Target.from_grade(5)
    harder = attempt(1, text=MEDIUM, gap=target.gap(measure(MEDIUM).grade))
    other = attempt(2, text=DECREE, gap=target.gap(measure(DECREE).grade))
    easier = attempt(3, gap=target.gap(measure(SIMPLE).grade))
    text = critique(easier, [other, harder, easier], measure(DECREE), target, FACTS)
    lines = text.splitlines()
    # Середина ступени 4-6 - класс 5: доля пути от попытки 3 (1.0) к попытке 1 (25.5)
    share = (5 - 1.0) / (25.5 - 1.0)
    words = SIMPLE_M.words_per_sent + share * (13.0 - SIMPLE_M.words_per_sent)
    letters = SIMPLE_M.letters_per_word + share * (9.08 - SIMPLE_M.letters_per_word)
    assert lines[4] == (
        "Попытка 1 сложнее цели, попытка 3 проще: нужен текст между ними, "
        f"около {words:.0f} слов в предложении и {letters:.1f} буквы в слове."
    )
    assert lines[5:] == [
        "Все попытки:",
        "2. Сводный класс 26.5: 18.0 слова в предложении, 8.33 буквы в слове, "
        "50% слов от пяти слогов",
        "1. Сводный класс 25.5: 13.0 слова в предложении, 9.08 буквы в слове, "
        "46% слов от пяти слогов",
        "3. Сводный класс 1.0: 4.3 слова в предложении, 5.23 буквы в слове, "
        "0% слов от пяти слогов",
    ]


def test_critique_facts():
    """На уровне цели - только факты: искаженные с цитатой и потерянные"""
    checks = [
        FactCheck(fact=1, status=FactStatus.DISTORTED, quote="правила о налогах"),
        FactCheck(fact=2, status=FactStatus.LOST, quote=""),
    ]
    current = attempt(2, checks=checks)
    text = critique(current, [current], measure(DECREE), TARGET, FACTS)
    assert text.splitlines() == [
        "Попытка 2 читается как «1-3-й класс (6-8 лет)» (сводный класс 1.0), "
        "цель - «1-3-й класс (6-8 лет)»: уровень достигнут, сохраните его.",
        "Искажены факты, верните их точный смысл:",
        "1. Постановление устанавливает порядок субсидий - в черновике: «правила о налогах»",
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
