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


def critique(attempt: Attempt, source: Measurement, target: Target, facts: list[str]) -> str:
    """
    Критика попытки без модели

    Описание:
        Указания из признаков ruTS, из которых складывается сводный класс,
        в сравнении с исходным текстом, и список искаженных и потерянных фактов
        с цитатами судьи. Сигнал не зависит от модели, которую он оценивает

    Аргументы:
        attempt (Attempt): Попытка
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
        lines.append(
            "Делите длинные предложения и заменяйте длинные и многосложные слова короткими."
            if attempt.gap > 0
            else "Объединяйте короткие предложения и используйте более точные и длинные слова."
        )
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
