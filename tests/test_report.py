import io
import math
from dataclasses import replace

from ruts_adapt.attempts import Attempt, Outcome, Result
from ruts_adapt.measure import Target, measure
from ruts_adapt.report import Reporter, table
from ruts_adapt.tasks import FactStatus
from tests.samples import DECREE, FACTS, KEPT, MEDIUM, SIMPLE, verdict

TARGET = Target.from_grade(2)
SOURCE = measure(DECREE)


def reporter():
    out = io.StringIO()
    return Reporter(TARGET, 3, out), out


def attempt(number, text, checks=KEPT.checks, tokens=330):
    measurement = measure(text)
    return Attempt(number, text, measurement, checks, TARGET.gap(measurement.grade), tokens)


def test_updates():
    report, out = reporter()
    report.update("measure_source", {"source_measurement": SOURCE})
    report.update("extract_facts", {"facts": FACTS, "facts_tokens": 165})
    report.update("rewrite", {"draft": MEDIUM, "draft_tokens": 165})
    report.update("judge", {"checks": verdict(FactStatus.KEPT, FactStatus.LOST).checks})
    report.update("measure", {"draft_measurement": measure(MEDIUM)})
    first = attempt(1, MEDIUM)
    report.update("record", {"attempts": [first]})
    report.update("critique", {"critique": "Строка 1\nСтрока 2"})
    report.update("finish", {})
    assert out.getvalue().splitlines() == [
        "Исходный текст: «аспирантура (старше 22 лет)», сводный класс 26.5, 18 слов; "
        "упрощаем до «1-3-й класс (6-8 лет)»",
        "Фактов в исходном тексте: 2",
        "Попытка 1: черновик готов",
        "Попытка 1: факты - сохранено 1, искажено 0, потеряно 1",
        "Попытка 1: «аспирантура (старше 22 лет)», сводный класс 25.5",
        "Критика:",
        "  Строка 1",
        "  Строка 2",
    ]
    assert report.attempts == [first]


def test_direction():
    report, out = reporter()
    report.target = Target.from_grade(17)
    report.update("measure_source", {"source_measurement": measure(SIMPLE)})
    assert out.getvalue().endswith("усложняем до «4-6-й курс вуза (20-22 года)»\n")


def test_on_level():
    report, out = reporter()
    report.update("measure_source", {"source_measurement": measure(SIMPLE)})
    report.finish(Result(Outcome.ON_LEVEL, SIMPLE, None))
    assert out.getvalue().splitlines() == [
        "Исходный текст: «1-3-й класс (6-8 лет)», сводный класс 1.0, 13 слов; "
        "уже на уровне цели «1-3-й класс (6-8 лет)»",
        "Текст уже на уровне цели, переписывать не нужно",
    ]


def test_accepted():
    report, out = reporter()
    report.source = SOURCE
    chosen = attempt(1, SIMPLE)
    report.attempts = [chosen]
    report.finish(Result(Outcome.ACCEPTED, SIMPLE, chosen), tokens=495)
    lines = out.getvalue().splitlines()
    assert lines[0] == ""
    assert lines[1].startswith("Попытка   Класс  Ступень")
    assert lines[-2:] == [
        "Готово: попытка 1 на уровне цели, все факты сохранены",
        "Токены всех вызовов модели: 495",
    ]


def test_best():
    """Лучшая попытка после лимита: причины, по которым она не принята"""
    report, out = reporter()
    report.source = SOURCE
    chosen = attempt(1, MEDIUM, verdict(FactStatus.DISTORTED, FactStatus.LOST).checks)
    report.attempts = [chosen]
    report.finish(Result(Outcome.BEST, MEDIUM, chosen))
    assert out.getvalue().splitlines()[-2] == (
        "Лимит попыток исчерпан, выбрана лучшая - попытка 1: уровень «аспирантура "
        "(старше 22 лет)» вместо «1-3-й класс (6-8 лет)», искажено или потеряно фактов: 2"
    )


def test_best_level_only():
    report, out = reporter()
    report.source = SOURCE
    chosen = attempt(1, MEDIUM)
    report.attempts = [chosen]
    report.finish(Result(Outcome.BEST, MEDIUM, chosen))
    assert out.getvalue().splitlines()[-2].endswith("вместо «1-3-й класс (6-8 лет)»")


def test_best_facts_only():
    report, out = reporter()
    report.source = SOURCE
    chosen = attempt(1, SIMPLE, verdict(FactStatus.KEPT, FactStatus.LOST).checks)
    report.attempts = [chosen]
    report.finish(Result(Outcome.BEST, SIMPLE, chosen))
    assert out.getvalue().splitlines()[-2].endswith("попытка 1: искажено или потеряно фактов: 1")


def test_mtld_warning():
    long_source = measure("Кот спит на окне, а пес лежит у двери. " * 20)
    text = "Кот спит. Кот спит. " * 50
    chosen = Attempt(1, text, measure(text), KEPT.checks, 0.0, 330)
    for source, warned in ((long_source, False), (replace(long_source, mtld=40.0), True)):
        report, out = reporter()
        report.source, report.attempts = source, [chosen]
        report.finish(Result(Outcome.ACCEPTED, text, chosen))
        last = out.getvalue().splitlines()[-1]
        assert last.startswith("Предупреждение: лексическое разнообразие (MTLD) упало на 75%") is (
            warned
        )


def test_table():
    first = attempt(1, MEDIUM, verdict(FactStatus.KEPT, FactStatus.LOST).checks)
    second = replace(attempt(2, SIMPLE), measurement=replace(measure(SIMPLE), mtld=math.inf))
    assert table(SOURCE, [first, second], second).splitlines() == [
        "Попытка   Класс  Ступень                      Слов в предл.  Слогов  Букв  "
        "От 5 слогов  MTLD  Факты с/и/п  Токены",
        "исходный  26.5   аспирантура (старше 22 лет)  18.0           3.61    8.33  "
        "50%          90.7  -            -",
        "1         25.5   аспирантура (старше 22 лет)  13.0           4.00    9.08  "
        "46%          -     1/0/1        330",
        "2 *       1.0    1-3-й класс (6-8 лет)        4.3            2.15    5.23  "
        "0%           -     2/0/0        330",
    ]
