import math
from dataclasses import dataclass

from ruts import LexicalStats

from ruts_adapt.attempts import Attempt, Outcome, Result
from ruts_adapt.measure import Measurement, Target
from ruts_adapt.tasks import FactStatus

OUTCOMES = {Outcome.ON_LEVEL: "уже на уровне", Outcome.ACCEPTED: "принят", Outcome.BEST: "лучшая"}


@dataclass(frozen=True)
class Run:
    """Живой прогон одного текста для отчета; result - None, если граф упал с ошибкой error"""

    case: str
    genre: str
    target: Target
    source: str
    source_measurement: Measurement
    facts: int
    attempts: list[Attempt]
    result: Result | None
    tokens: int
    error: str | None = None


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
        if run.result is not None:
            chosen = run.result.attempt
            outcome = OUTCOMES[run.result.outcome] + (
                f", попытка {chosen.number}" if chosen else ""
            )
            final_text = run.result.text
        else:
            chosen = run.attempts[-1] if run.attempts else None
            outcome = f"ошибка: {run.error}"
            final_text = chosen.text if chosen else run.source
        final = chosen.measurement if chosen else source
        grades = " → ".join(f"{attempt.measurement.grade:.1f}" for attempt in run.attempts)
        facts = " → ".join(
            "/".join(
                str(sum(check.status is status for check in attempt.checks))
                for status in FactStatus
            )
            for attempt in run.attempts
        )
        lines.append(
            f"| {run.case} | {run.genre} | {run.target.level} | {source.grade:.1f} "
            f"| {grades or '-'} | {facts or '-'} (из {run.facts}) "
            f"| {outcome} "
            f"| {source.words_per_sent:.1f} → {final.words_per_sent:.1f} "
            f"| {rare_words(run.source):.0%} → {rare_words(final_text):.0%} "
            f"| {_mtld(source.mtld)} → {_mtld(final.mtld)} | {run.tokens} |"
        )
    return "\n".join(lines)


def _mtld(value: float) -> str:
    """MTLD для таблицы: inf, когда в тексте нет повторяющихся слов, - прочерк, как в команде"""
    return f"{value:.0f}" if math.isfinite(value) else "-"
