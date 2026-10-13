from dataclasses import dataclass

from ruts import LexicalStats

from ruts_adapt.attempts import Attempt, Outcome, Result
from ruts_adapt.measure import Measurement, Target
from ruts_adapt.tasks import FactStatus

OUTCOMES = {Outcome.ON_LEVEL: "уже на уровне", Outcome.ACCEPTED: "принят", Outcome.BEST: "лучшая"}


@dataclass(frozen=True)
class Run:
    """Живой прогон одного текста для отчета"""

    case: str
    genre: str
    target: Target
    source: str
    source_measurement: Measurement
    facts: int
    attempts: list[Attempt]
    result: Result
    tokens: int


def rare_words(text: str) -> float:
    """Доля слов вне топ-10000 частотного списка Шарова: термины и редкие слова"""
    return 1 - LexicalStats(text).band_coverage()[10000]


def report(runs: list[Run]) -> str:
    """Таблица живых прогонов в Markdown: траектория класса и фактов по попыткам, итог"""
    lines = [
        "| Текст | Жанр | Цель | Исходник | Класс по попыткам | Факты с/и/п по попыткам "
        "| Итог | Слов в предл. | Слов вне топ-10000 | MTLD | Токены |",
        "| :-- | :-- | :-- | :-: | :-- | :-- | :-- | :-: | :-: | :-: | :-: |",
    ]
    for run in runs:
        source = run.source_measurement
        final = run.result.attempt.measurement if run.result.attempt else source
        grades = " → ".join(f"{attempt.measurement.grade:.1f}" for attempt in run.attempts)
        facts = " → ".join(
            "/".join(
                str(sum(check.status is status for check in attempt.checks))
                for status in FactStatus
            )
            for attempt in run.attempts
        )
        chosen = f", попытка {run.result.attempt.number}" if run.result.attempt else ""
        lines.append(
            f"| {run.case} | {run.genre} | {run.target.level} | {source.grade:.1f} "
            f"| {grades or '-'} | {facts or '-'} (из {run.facts}) "
            f"| {OUTCOMES[run.result.outcome]}{chosen} "
            f"| {source.words_per_sent:.1f} → {final.words_per_sent:.1f} "
            f"| {rare_words(run.source):.0%} → {rare_words(run.result.text):.0%} "
            f"| {source.mtld:.0f} → {final.mtld:.0f} | {run.tokens} |"
        )
    return "\n".join(lines)
