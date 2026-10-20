import math
from dataclasses import dataclass
from enum import StrEnum

from ruts.constants import LEVEL_NORMS

from .measure import Measurement, Target, number
from .tasks import FactCheck, FactStatus

MTLD_DROP = 0.3
MTLD_MIN_WORDS = 100
# Ступени без норм: предложение короткое до верхней границы нормы 4-6, длинное - от нее же у 10-11
SHORT_SENTENCE = LEVEL_NORMS[4]["words_per_sent"][1]
LONG_SENTENCE = LEVEL_NORMS[10]["words_per_sent"][1]


@dataclass(frozen=True)
class Attempt:
    """
    Попытка адаптации и ее оценка

    Атрибуты:
        number (int): Номер попытки, с 1
        text (str): Черновик
        measurement (Measurement): Уровень и признаки черновика
        checks (list[FactCheck]): Проверки фактов исходника в черновике
        gap (float): Расстояние сводного класса до ступени цели (Target.gap)
        tokens (int): Токены писателя и судьи на эту попытку, без повторов узлов
    """

    number: int
    text: str
    measurement: Measurement
    checks: list[FactCheck]
    gap: float
    tokens: int

    @property
    def lost(self) -> list[FactCheck]:
        """Потерянные факты"""
        return [check for check in self.checks if check.status is FactStatus.LOST]

    @property
    def distorted(self) -> list[FactCheck]:
        """Искаженные факты"""
        return [check for check in self.checks if check.status is FactStatus.DISTORTED]

    @property
    def accepted(self) -> bool:
        """Текст на уровне цели, и все факты сохранены"""
        return self.gap == 0 and not self.lost and not self.distorted


class Outcome(StrEnum):
    """Чем закончилась адаптация"""

    ON_LEVEL = "on_level"
    ACCEPTED = "accepted"
    BEST = "best"


@dataclass(frozen=True)
class Result:
    """
    Итог адаптации

    Атрибуты:
        outcome (Outcome): Исходник уже на уровне, попытка принята или выбрана
            лучшая после исчерпания попыток
        text (str): Итоговый текст: исходник или черновик попытки
        attempt (Attempt | None): Выбранная попытка; None, если исходник уже на уровне
    """

    outcome: Outcome
    text: str
    attempt: Attempt | None


def best(attempts: list[Attempt]) -> Attempt:
    """
    Лучшая попытка, если ни одна не принята

    Описание:
        Сначала сохранность смысла, затем уровень: меньше искаженных и потерянных
        фактов, затем ближе к ступени цели, при равенстве - более ранняя попытка

    Аргументы:
        attempts (list[Attempt]): Попытки, хотя бы одна

    Вывод:
        Attempt: Лучшая попытка
    """
    return min(
        attempts,
        key=lambda attempt: (
            len(attempt.lost) + len(attempt.distorted),
            abs(attempt.gap),
            attempt.number,
        ),
    )


def candidate(attempts: list[Attempt], after: int = 0) -> Result:
    """
    Итог по попыткам

    Описание:
        Последняя попытка, если она принята, иначе лучшая (best) из попыток
        с номером больше after. После возврата проверяющим after - число попыток
        в момент возврата: более ранние, в том числе возвращенные, написаны
        без его замечаний и не выбираются

    Аргументы:
        attempts (list[Attempt]): Попытки, хотя бы одна с номером больше after
        after (int): Номер последней попытки, которая не может стать итогом

    Вывод:
        Result: Принятая или лучшая попытка
    """
    remaining = [attempt for attempt in attempts if attempt.number > after]
    if remaining[-1].accepted:
        return Result(Outcome.ACCEPTED, remaining[-1].text, remaining[-1])
    chosen = best(remaining)
    return Result(Outcome.BEST, chosen.text, chosen)


def critique(
    attempt: Attempt,
    attempts: list[Attempt],
    source: Measurement,
    target: Target,
    facts: list[str],
) -> str:
    """
    Критика попытки без модели

    Описание:
        Указания из признаков ruTS, из которых складывается сводный класс:
        в сравнении с исходным текстом, с нормами ступени цели (Target.norm)
        и с другими попытками. Совет, что менять - слова или предложения, - учитывает
        нормы ступени, а без них - крайние длины предложения (SHORT_SENTENCE,
        LONG_SENTENCE). Для ступени без норм, если одна попытка сложнее цели,
        а другая проще, критика вместо совета называет признаки текста между
        ними - линейно по сводному классу к середине ступени. Затем список
        искаженных и потерянных фактов с цитатами судьи. Сигнал не зависит
        от модели, которую он оценивает. Числа - с десятичной запятой

    Аргументы:
        attempt (Attempt): Попытка, к которой критика
        attempts (list[Attempt]): Все попытки по порядку
        source (Measurement): Уровень и признаки исходного текста
        target (Target): Цель адаптации
        facts (list[str]): Факты исходного текста

    Вывод:
        str: Критика для следующей попытки
    """
    current = attempt.measurement
    level = (
        f"Попытка {attempt.number} читается как «{current.level}» "
        f"(сводный класс {number(current.grade)}), цель - «{target.level}»"
    )
    lines = []
    if attempt.gap > 0:
        lines.append(f"{level}: текст сложнее цели на {number(attempt.gap)} класса.")
    elif attempt.gap < 0:
        lines.append(f"{level}: текст проще цели на {number(-attempt.gap)} класса.")
    else:
        lines.append(f"{level}: уровень достигнут, сохраните его.")
    if attempt.gap != 0:
        lines.append(
            f"Сейчас в среднем {number(current.words_per_sent)} слова в предложении "
            f"(в исходном {number(source.words_per_sent)}), "
            f"{number(current.syllables_per_word, 2)} слога и "
            f"{number(current.letters_per_word, 2)} буквы в слове "
            f"({number(source.syllables_per_word, 2)} и {number(source.letters_per_word, 2)}), "
            f"{current.complex_words:.0%} слов от пяти слогов ({source.complex_words:.0%})."
        )
        norm = target.describe_norm()
        if norm is not None:
            lines.append(f"В текстах для этого уровня {norm}.")
        # С нормами ориентир дают они: признаки между попытками могли бы им противоречить
        between = None if target.norm else _between(attempts, target)
        lines.append(between or _advice(attempt, target))
    if len(attempts) > 1:
        lines.append("Все попытки:")
        lines += [
            f"{item.number}. Сводный класс {number(item.measurement.grade)}: "
            f"{number(item.measurement.words_per_sent)} слова в предложении, "
            f"{number(item.measurement.letters_per_word, 2)} буквы в слове, "
            f"{item.measurement.complex_words:.0%} слов от пяти слогов"
            for item in attempts
        ]
    if attempt.distorted:
        lines.append("Искажены факты, верните их точный смысл:")
        lines += [
            f"{check.fact}. {facts[check.fact - 1]} - в черновике: «{check.quote}»"
            for check in attempt.distorted
        ]
    if attempt.lost:
        lines.append("Потеряны факты, верните их:")
        lines += [f"{check.fact}. {facts[check.fact - 1]}" for check in attempt.lost]
    return "\n".join(lines)


def _advice(attempt: Attempt, target: Target) -> str:
    """Что менять в тексте не на уровне: слова или предложения, с оглядкой на нормы ступени"""
    words = attempt.measurement.words_per_sent
    norm = target.norm
    if attempt.gap > 0:
        if words <= (norm["words_per_sent"][1] if norm else SHORT_SENTENCE):
            already = "не длиннее, чем в текстах этого уровня" if norm else "короткие"
            return (
                f"Предложения уже {already}: не дробите их дальше, "
                "а заменяйте длинные слова и термины короткими и простыми."
            )
        if (
            norm is not None
            and attempt.measurement.letters_per_word <= norm["letters_per_word"][1]
        ):
            return "Слова уже не длиннее, чем в текстах этого уровня: делите длинные предложения."
        return "Делите длинные предложения и заменяйте длинные и многосложные слова короткими."
    if norm is not None and words < norm["words_per_sent"][0]:
        return "Предложения короче, чем в текстах этого уровня: объединяйте их."
    if norm is not None or words >= LONG_SENTENCE:
        already = "не короче, чем в текстах этого уровня" if norm else "длинные"
        return (
            f"Предложения уже {already}: не объединяйте их дальше, "
            "а используйте более точные и длинные слова."
        )
    return "Объединяйте короткие предложения и используйте более точные и длинные слова."


def _between(attempts: list[Attempt], target: Target) -> str | None:
    """Признаки текста между ближайшими к цели попытками по обе стороны ступени"""
    above = [attempt for attempt in attempts if attempt.gap > 0]
    below = [attempt for attempt in attempts if attempt.gap < 0]
    if not above or not below:
        return None
    harder = min(above, key=lambda attempt: attempt.gap)
    easier = max(below, key=lambda attempt: attempt.gap)
    high, low = harder.measurement, easier.measurement
    share = ((target.low + target.high) / 2 - low.grade) / (high.grade - low.grade)
    words = low.words_per_sent + share * (high.words_per_sent - low.words_per_sent)
    letters = low.letters_per_word + share * (high.letters_per_word - low.letters_per_word)
    return (
        f"Попытка {harder.number} сложнее цели, попытка {easier.number} проще: нужен текст "
        f"между ними, около {number(words, 0)} слов в предложении и {number(letters)} буквы "
        "в слове."
    )


def mtld_drop(source: Measurement, result: Measurement) -> float | None:
    """
    Падение лексического разнообразия MTLD относительно исходника

    Аргументы:
        source (Measurement): Уровень и признаки исходного текста
        result (Measurement): Уровень и признаки итогового текста

    Вывод:
        float | None: Доля падения от 0 до 1; None, если MTLD не упал, не определен
            (inf, когда в тексте нет повторяющихся слов) или в одном из текстов меньше
            100 слов - на коротком тексте MTLD неустойчив

    Пример использования:
        >>> from ruts_adapt.measure import measure
        >>> source = measure("Кот спит на окне, а пес лежит у двери. " * 20)
        >>> round(mtld_drop(source, measure("Кот спит. Кот спит. " * 50)), 2)
        0.24
    """
    if min(source.n_words, result.n_words) < MTLD_MIN_WORDS:
        return None
    if not (math.isfinite(source.mtld) and math.isfinite(result.mtld)) or source.mtld <= 0:
        return None
    drop = 1 - result.mtld / source.mtld
    return drop if drop > 0 else None
