import math
from typing import Any, TextIO

from .attempts import MTLD_DROP, Attempt, Outcome, Result, mtld_drop
from .measure import Direction, Measurement, Target
from .tasks import FactCheck, FactStatus

DIRECTIONS = {
    Direction.SIMPLIFY: "упрощаем до",
    Direction.COMPLICATE: "усложняем до",
    Direction.KEEP: "уже на уровне цели",
}
COLUMNS = (
    "Попытка",
    "Класс",
    "Ступень",
    "Слов в предл.",
    "Слогов",
    "Букв",
    "От 5 слогов",
    "MTLD",
    "Факты с/и/п",
    "Токены",
)


class Reporter:
    """
    Ход адаптации по обновлениям узлов графа

    Описание:
        Получает обновления потока графа (stream_mode="updates") и печатает
        строку на каждый шаг, а в конце - таблицу попыток с метриками ruTS,
        итог и предупреждение о падении лексического разнообразия

    Аргументы:
        target (Target): Цель адаптации
        max_attempts (int): Наибольшее число попыток
        out (TextIO): Куда печатать

    Атрибуты:
        source (Measurement | None): Уровень и признаки исходника
        attempts (list[Attempt]): Попытки в порядке появления
    """

    def __init__(self, target: Target, max_attempts: int, out: TextIO) -> None:
        self.target = target
        self.max_attempts = max_attempts
        self.out = out
        self.source: Measurement | None = None
        self.attempts: list[Attempt] = []
        self._number = 0

    def update(self, node: str, values: dict[str, Any]) -> None:
        """
        Строка о шаге графа

        Аргументы:
            node (str): Узел, который закончил работу
            values (dict[str, Any]): Его обновление состояния
        """
        if node == "measure_source":
            self.source = values["source_measurement"]
            direction = self.target.direction(self.source.grade)
            self._print(
                f"Исходный текст: «{self.source.level}», сводный класс {self.source.grade:.1f}, "
                f"{self.source.n_words} слов; {DIRECTIONS[direction]} «{self.target.level}»"
            )
        elif node == "extract_facts":
            self._print(f"Фактов в исходном тексте: {len(values['facts'])}")
        elif node == "rewrite":
            self._number += 1
            self._print(f"Попытка {self._number}: черновик готов")
        elif node == "measure":
            current = values["draft_measurement"]
            self._print(
                f"Попытка {self._number}: «{current.level}», сводный класс {current.grade:.1f}"
            )
        elif node == "judge":
            kept, distorted, lost = _counts(values["checks"])
            self._print(
                f"Попытка {self._number}: факты - сохранено {kept}, искажено {distorted}, "
                f"потеряно {lost}"
            )
        elif node == "record":
            self.attempts.append(values["attempts"][0])
        elif node == "critique":
            self._print("Критика:")
            for line in values["critique"].splitlines():
                self._print(f"  {line}")

    def finish(self, result: Result, tokens: int = 0) -> None:
        """
        Итог, таблица попыток и предупреждения

        Аргументы:
            result (Result): Итог адаптации
            tokens (int): Токены всех вызовов модели, включая повторы
        """
        if result.outcome is Outcome.ON_LEVEL:
            self._print("Текст уже на уровне цели, переписывать не нужно")
            return
        chosen = result.attempt
        assert chosen is not None and self.source is not None
        self._print("")
        self._print(table(self.source, self.attempts, chosen))
        self._print("")
        if result.outcome is Outcome.ACCEPTED:
            self._print(f"Готово: попытка {chosen.number} на уровне цели, все факты сохранены")
        else:
            problems = len(chosen.lost) + len(chosen.distorted)
            reasons = []
            if chosen.gap != 0:
                reasons.append(
                    f"уровень «{chosen.measurement.level}» вместо «{self.target.level}»"
                )
            if problems:
                reasons.append(f"искажено или потеряно фактов: {problems}")
            self._print(
                f"Лимит попыток исчерпан, выбрана лучшая - попытка {chosen.number}: "
                + ", ".join(reasons)
            )
        self._print(f"Токены всех вызовов модели: {tokens}")
        drop = mtld_drop(self.source, chosen.measurement)
        if drop is not None and drop > MTLD_DROP:
            self._print(
                f"Предупреждение: лексическое разнообразие (MTLD) упало на {drop:.0%} - "
                "текст мог стать однообразным"
            )

    def _print(self, line: str) -> None:
        print(line, file=self.out, flush=True)


def table(source: Measurement, attempts: list[Attempt], chosen: Attempt) -> str:
    """
    Таблица попыток с метриками ruTS

    Аргументы:
        source (Measurement): Уровень и признаки исходника
        attempts (list[Attempt]): Попытки
        chosen (Attempt): Выбранная попытка, отмечается звездочкой

    Вывод:
        str: Таблица с выравниванием по столбцам
    """
    rows = [list(COLUMNS), ["исходный", *_metrics(source), "-", "-"]]
    for attempt in attempts:
        mark = " *" if attempt is chosen else ""
        rows.append(
            [
                f"{attempt.number}{mark}",
                *_metrics(attempt.measurement),
                "/".join(map(str, _counts(attempt.checks))),
                str(attempt.tokens),
            ]
        )
    widths = [max(len(row[column]) for row in rows) for column in range(len(COLUMNS))]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in rows
    )


def _metrics(measurement: Measurement) -> list[str]:
    """Ячейки метрик ruTS"""
    mtld = f"{measurement.mtld:.1f}" if math.isfinite(measurement.mtld) else "-"
    return [
        f"{measurement.grade:.1f}",
        measurement.level,
        f"{measurement.words_per_sent:.1f}",
        f"{measurement.syllables_per_word:.2f}",
        f"{measurement.letters_per_word:.2f}",
        f"{measurement.complex_words:.0%}",
        mtld,
    ]


def _counts(checks: list[FactCheck]) -> tuple[int, int, int]:
    """Сохраненные, искаженные и потерянные факты"""
    statuses = [check.status for check in checks]
    return (
        statuses.count(FactStatus.KEPT),
        statuses.count(FactStatus.DISTORTED),
        statuses.count(FactStatus.LOST),
    )
