import math
from collections import Counter
from dataclasses import dataclass

from ruts import LexicalStats

from ruts_adapt.attempts import Attempt, Outcome, Result
from ruts_adapt.measure import Measurement, Target
from ruts_adapt.tasks import FactCheck, FactStatus

OUTCOMES = {Outcome.ON_LEVEL: "уже на уровне", Outcome.ACCEPTED: "принят", Outcome.BEST: "лучшая"}


@dataclass(frozen=True)
class Run:
    """
    Живой прогон графа на одном тексте

    Атрибуты:
        critiques (list[str]): Критика попыток по порядку: первая - к попытке 1
        result (Result): Итог; None, если граф упал с ошибкой error
    """

    case: str
    genre: str
    target: Target
    source: str
    source_measurement: Measurement
    facts: list[str]
    attempts: list[Attempt]
    critiques: list[str]
    result: Result | None
    tokens: int
    error: str | None = None


@dataclass(frozen=True)
class JudgeRun:
    """
    Прогон стенда судьи на одном тексте

    Атрибуты:
        plan (dict[int, FactStatus]): Порча по номерам фактов: убран или искажен
        text (str): Испорченный пересказ; None, если писатель не ответил
        changes (list[str]): Что писатель изменил, его словами
        checks (list[FactCheck]): Вердикт судьи; None, если стенд упал с ошибкой error
    """

    case: str
    facts: list[str]
    plan: dict[int, FactStatus]
    text: str | None
    changes: list[str]
    checks: list[FactCheck] | None
    tokens: int
    error: str | None = None

    def matrix(self) -> Counter[tuple[FactStatus, FactStatus]]:
        """Число фактов по парам (что сделал писатель, что решил судья)"""
        if self.checks is None:
            return Counter()
        return Counter(
            (self.plan.get(check.fact, FactStatus.KEPT), check.status) for check in self.checks
        )


def rare_words(text: str) -> float:
    """Доля слов вне топ-10000 частотного списка Шарова: термины и редкие слова"""
    return 1 - LexicalStats(text).band_coverage()[10000]


def summary(runs: list[Run], judge_runs: list[JudgeRun]) -> str:
    """Таблицы прогонов графа и стенда судьи в Markdown - для терминала и начала отчета"""
    sections = []
    if runs:
        sections.append(f"## Прогоны графа\n\n{_runs_table(runs)}")
    if judge_runs:
        sections.append(f"## Стенд судьи\n\n{_judge_table(judge_runs)}")
    return "\n\n".join(sections)


def details(runs: list[Run], judge_runs: list[JudgeRun]) -> str:
    """Тексты прогонов в Markdown: исходники, факты, черновики, критика, расхождения судьи"""
    sections = []
    if runs:
        sections.append("## Тексты прогонов графа")
        sections += [_run_texts(run) for run in runs]
    if judge_runs:
        sections.append("## Тексты стенда судьи")
        sections += [_judge_texts(run) for run in judge_runs]
    return "\n\n".join(sections)


def _runs_table(runs: list[Run]) -> str:
    """Траектория класса и фактов по попыткам, итог"""
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
        facts = " → ".join(_statuses(attempt.checks) for attempt in run.attempts)
        lines.append(
            f"| {run.case} | {run.genre} | {run.target.level} | {source.grade:.1f} "
            f"| {grades or '-'} | {facts or '-'} (из {len(run.facts)}) "
            f"| {outcome} "
            f"| {source.words_per_sent:.1f} → {final.words_per_sent:.1f} "
            f"| {rare_words(run.source):.0%} → {rare_words(final_text):.0%} "
            f"| {_mtld(source.mtld)} → {_mtld(final.mtld)} | {run.tokens} |"
        )
    return "\n".join(lines)


def _judge_table(runs: list[JudgeRun]) -> str:
    """Вердикты судьи с/и/п по тому, что писатель сделал с фактом, и итог по всем текстам"""
    lines = [
        "| Текст | Фактов | Убраны: с/и/п | Искажены: с/и/п | Не тронуты: с/и/п | Токены |",
        "| :-- | :-: | :-: | :-: | :-: | :-: |",
    ]
    total: Counter[tuple[FactStatus, FactStatus]] = Counter()
    for run in runs:
        cells = (
            _matrix_cells(run.matrix()) if run.checks is not None else f"ошибка: {run.error} | | "
        )
        total += run.matrix()
        lines.append(f"| {run.case} | {len(run.facts)} | {cells} | {run.tokens} |")
    lines.append(f"| Всего | | {_matrix_cells(total)} | {sum(run.tokens for run in runs)} |")
    damaged = [FactStatus.LOST, FactStatus.DISTORTED]
    noticed = sum(total[truth, verdict] for truth in damaged for verdict in damaged)
    exact = sum(total[truth, truth] for truth in damaged)
    n_damaged = sum(total[truth, verdict] for truth in damaged for verdict in FactStatus)
    alarms = sum(total[FactStatus.KEPT, verdict] for verdict in damaged)
    n_kept = sum(total[FactStatus.KEPT, verdict] for verdict in FactStatus)
    lines += [
        "",
        f"Судья заметил порчу: {_share(noticed, n_damaged)}, с верным статусом: "
        f"{_share(exact, n_damaged)}; ложные тревоги: {_share(alarms, n_kept)} нетронутых фактов",
    ]
    return "\n".join(lines)


def _run_texts(run: Run) -> str:
    """Исходник, факты и попытки одного прогона графа с вердиктами и критикой"""
    source = run.source_measurement
    parts = [
        f"### {run.case}: {run.target.level}",
        f"Исходник: класс {source.grade:.1f}, {source.words_per_sent:.1f} слова в предложении",
        _paragraphs(run.source),
    ]
    if run.facts:
        parts.append(_numbered(run.facts))
    for attempt in run.attempts:
        parts += [
            f"#### Попытка {attempt.number}: класс {attempt.measurement.grade:.1f}, "
            f"{attempt.measurement.words_per_sent:.1f} слова в предложении, "
            f"факты {_statuses(attempt.checks)}",
            _paragraphs(attempt.text),
        ]
        problems = [
            f"- {check.fact}. {run.facts[check.fact - 1]} - {check.status}{_quote(check.quote)}"
            for check in attempt.checks
            if check.status is not FactStatus.KEPT
        ]
        if problems:
            parts.append("\n".join(problems))
        if attempt.number <= len(run.critiques):
            parts.append(f"Критика:\n\n```text\n{run.critiques[attempt.number - 1]}\n```")
    if run.error:
        parts.append(f"Ошибка: {run.error}")
    return "\n\n".join(parts)


def _judge_texts(run: JudgeRun) -> str:
    """Испорченный пересказ, порча со слов писателя и расхождения судьи с ней"""
    parts = [f"### {run.case}"]
    if run.text is not None:
        parts += [_paragraphs(run.text), "Порча со слов писателя:"]
        parts.append("\n".join(f"- {change}" for change in run.changes) or "-")
    if run.checks is not None:
        misses = [
            f"- {check.fact}. {run.facts[check.fact - 1]} - писатель: "
            f"{run.plan.get(check.fact, FactStatus.KEPT)}, судья: {check.status}"
            f"{_quote(check.quote)}"
            for check in run.checks
            if check.status is not run.plan.get(check.fact, FactStatus.KEPT)
        ]
        parts.append("Расхождения судьи с порчей:")
        parts.append("\n".join(misses) or "нет")
    if run.error:
        parts.append(f"Ошибка: {run.error}")
    return "\n\n".join(parts)


def _statuses(checks: list[FactCheck]) -> str:
    """Число сохраненных, искаженных и потерянных фактов: с/и/п"""
    return "/".join(str(sum(check.status is status for check in checks)) for status in FactStatus)


def _matrix_cells(matrix: Counter[tuple[FactStatus, FactStatus]]) -> str:
    """Ячейки с/и/п для убранных, искаженных и нетронутых фактов"""
    return " | ".join(
        "/".join(str(matrix[truth, verdict]) for verdict in FactStatus)
        for truth in (FactStatus.LOST, FactStatus.DISTORTED, FactStatus.KEPT)
    )


def _share(part: int, whole: int) -> str:
    return f"{part} из {whole} ({part / whole:.0%})" if whole else "0 из 0"


def _paragraphs(text: str) -> str:
    """Абзацы текста через пустую строку, как их показывает Markdown"""
    return "\n\n".join(line.strip() for line in text.splitlines() if line.strip())


def _numbered(items: list[str]) -> str:
    return "\n".join(f"{number}. {item}" for number, item in enumerate(items, 1))


def _quote(quote: str) -> str:
    """Цитата судьи после вердикта; у потерянного факта ее нет"""
    return f": «{quote}»" if quote else ""


def _mtld(value: float) -> str:
    """MTLD для таблицы: inf, когда в тексте нет повторяющихся слов, - прочерк, как в команде"""
    return f"{value:.0f}" if math.isfinite(value) else "-"
