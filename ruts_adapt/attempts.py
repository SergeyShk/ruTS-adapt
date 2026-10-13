import math
from dataclasses import dataclass
from enum import StrEnum

from .measure import Measurement, Target
from .tasks import FactCheck, FactStatus

MTLD_DROP = 0.3
MTLD_MIN_WORDS = 100


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


def candidate(attempts: list[Attempt], rejected: list[int] | None = None) -> Result:
    """
    Итог по попыткам

    Описание:
        Последняя попытка, если она принята, иначе лучшая из всех (best);
        попытки, которые вернул проверяющий, не выбираются

    Аргументы:
        attempts (list[Attempt]): Попытки, хотя бы одна не возвращенная
        rejected (list[int]): Номера возвращенных попыток

    Вывод:
        Result: Принятая или лучшая попытка
    """
    rejected = rejected or []
    remaining = [attempt for attempt in attempts if attempt.number not in rejected]
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
        и с другими попытками. Для ступени без норм, если одна попытка сложнее
        цели, а другая проще, критика называет признаки текста между ними -
        линейно по сводному классу к середине ступени. Затем список искаженных
        и потерянных фактов с цитатами судьи. Сигнал не зависит от модели,
        которую он оценивает

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
        f"(сводный класс {current.grade:.1f}), цель - «{target.level}»"
    )
    lines = []
    if attempt.gap > 0:
        lines.append(f"{level}: текст сложнее цели на {attempt.gap:.1f} класса.")
    elif attempt.gap < 0:
        lines.append(f"{level}: текст проще цели на {-attempt.gap:.1f} класса.")
    else:
        lines.append(f"{level}: уровень достигнут, сохраните его.")
    if attempt.gap != 0:
        lines.append(
            f"Сейчас в среднем {current.words_per_sent:.1f} слова в предложении "
            f"(в исходном {source.words_per_sent:.1f}), {current.syllables_per_word:.2f} слога "
            f"и {current.letters_per_word:.2f} буквы в слове ({source.syllables_per_word:.2f} "
            f"и {source.letters_per_word:.2f}), {current.complex_words:.0%} слов от пяти слогов "
            f"({source.complex_words:.0%})."
        )
        if target.norm is not None:
            lines.append(f"В текстах для этого уровня {target.norm.describe()}.")
        lines.append(_advice(attempt, target))
        # С нормами ориентир дают они: признаки между попытками могли бы им противоречить
        if target.norm is None and (between := _between(attempts, target)):
            lines.append(between)
    if len(attempts) > 1:
        lines.append("Все попытки:")
        lines += [
            f"{item.number}. Сводный класс {item.measurement.grade:.1f}: "
            f"{item.measurement.words_per_sent:.1f} слова в предложении, "
            f"{item.measurement.letters_per_word:.2f} буквы в слове, "
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
        if norm is not None and words <= norm.words_per_sent[1]:
            return (
                "Предложения уже не длиннее, чем в текстах этого уровня: не дробите их дальше, "
                "а заменяйте длинные слова и термины короткими и простыми."
            )
        return "Делите длинные предложения и заменяйте длинные и многосложные слова короткими."
    if norm is None:
        return "Объединяйте короткие предложения и используйте более точные и длинные слова."
    if words < norm.words_per_sent[0]:
        return "Предложения короче, чем в текстах этого уровня: объединяйте их."
    return (
        "Предложения уже не короче, чем в текстах этого уровня: не объединяйте их дальше, "
        "а используйте более точные и длинные слова."
    )


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
        f"между ними, около {words:.0f} слов в предложении и {letters:.1f} буквы в слове."
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
