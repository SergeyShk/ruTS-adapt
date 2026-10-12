from dataclasses import dataclass
from enum import StrEnum

from ruts import BasicStats, DiversityStats, ReadabilityStats
from ruts.constants import GRADE_AGE_LEVELS
from ruts.readability_stats import grade_to_age

MIN_GRADE = GRADE_AGE_LEVELS[0][0]
MAX_GRADE = GRADE_AGE_LEVELS[-1][1]


class Direction(StrEnum):
    """Направление адаптации: упростить, усложнить или оставить текст на уровне"""

    SIMPLIFY = "simplify"
    COMPLICATE = "complicate"
    KEEP = "keep"


@dataclass(frozen=True)
class Measurement:
    """
    Уровень текста и его признаки по ruTS

    Описание:
        Уровень - сводный класс ruTS (медиана формул класса и индекса Флеша)
        и его прочтение как ступени обучения и возраста читателя; признаки -
        то, на что опираются формулы: длина предложений, доли сложных и длинных
        слов. Сложное слово - от 4 слогов, длинное - от 6 букв, как в BasicStats

    Атрибуты:
        grade (float): Сводный класс
        level (str): Ступень обучения и возраст читателя по сводному классу
        n_words (int): Количество слов
        n_sents (int): Количество предложений
        words_per_sent (float): Средняя длина предложения в словах
        complex_words (float): Доля сложных слов
        long_words (float): Доля длинных слов
        mtld (float): Лексическое разнообразие MTLD; inf, если текст так короток,
            что TTR ни разу не опускается до порога
    """

    grade: float
    level: str
    n_words: int
    n_sents: int
    words_per_sent: float
    complex_words: float
    long_words: float
    mtld: float


def measure(text: str) -> Measurement:
    """
    Измерение уровня и признаков текста

    Аргументы:
        text (str): Текст

    Вывод:
        Measurement: Уровень и признаки текста

    Исключения:
        SourceError: Если в тексте нет слов

    Пример использования:
        >>> m = measure("Существуют три вида лжи: ложь, наглая ложь и статистика")
        >>> m.grade, m.level
        (2.5, '1-3-й класс (6-8 лет)')
        >>> m.words_per_sent, round(m.complex_words, 2), round(m.long_words, 2)
        (9.0, 0.22, 0.33)
    """
    bs = BasicStats(text, normalize=True)
    rs = ReadabilityStats(bs)
    return Measurement(
        grade=rs.consensus_grade,
        level=rs.describe_grade(),
        n_words=bs.n_words,
        n_sents=bs.n_sents,
        words_per_sent=bs.n_words / bs.n_sents,
        complex_words=bs.p_complex_words,
        long_words=bs.p_long_words,
        mtld=DiversityStats(text).mtld,
    )


@dataclass(frozen=True)
class Target:
    """
    Цель адаптации - ступень обучения читателя

    Описание:
        Класс читателя относится к своей ступени таблицы ruTS GRADE_AGE_LEVELS:
        1-3, 4-6, 7-9, 10-11-й класс, 1-3 и 4-6-й курс вуза (годы обучения 12-17).
        Текст на уровне, если ruTS прочитывает его сводный класс как эту ступень

    Атрибуты:
        grade (int): Класс читателя
        low (int): Первый класс ступени
        high (int): Последний класс ступени
        level (str): Ступень обучения и возраст читателя

    Пример использования:
        >>> target = Target.from_grade(5)
        >>> target.level
        '4-6-й класс (9-11 лет)'
        >>> target.direction(8.5), target.direction(6.0), target.direction(2.0)
        (<Direction.SIMPLIFY: 'simplify'>, <Direction.KEEP: 'keep'>, <Direction.COMPLICATE: 'complicate'>)
    """

    grade: int
    low: int
    high: int
    level: str

    @classmethod
    def from_grade(cls, grade: int) -> "Target":
        """
        Цель по классу читателя

        Аргументы:
            grade (int): Класс школы 1-11 или год обучения 12-17

        Вывод:
            Target: Ступень обучения, к которой относится класс

        Исключения:
            ValueError: Если класс не целое число от 1 до 17
        """
        if isinstance(grade, int) and not isinstance(grade, bool):
            for low, high, _, _ in GRADE_AGE_LEVELS:
                if low <= grade <= high:
                    return cls(grade, low, high, grade_to_age(grade))
        raise ValueError(
            f"Класс читателя должен быть целым числом от {MIN_GRADE} до {MAX_GRADE}, "
            f"получено {grade!r}"
        )

    def gap(self, grade: float) -> float:
        """
        Расстояние сводного класса до ступени цели

        Описание:
            0, если текст на уровне; иначе разница с ближайшей границей ступени:
            больше 0 - текст сложнее цели, меньше 0 - проще. Сводный класс меньше 1
            относится к первой ступени, как в ruTS

        Аргументы:
            grade (float): Сводный класс текста

        Вывод:
            float: Расстояние в классах

        Пример использования:
            >>> target = Target.from_grade(5)
            >>> target.gap(6.0), target.gap(8.5), target.gap(2.0)
            (0.0, 2.5, -2.0)
        """
        if grade_to_age(grade) == self.level:
            return 0.0
        return grade - self.high if grade > self.high else grade - self.low

    def direction(self, grade: float) -> Direction:
        """
        Направление адаптации текста к цели

        Аргументы:
            grade (float): Сводный класс текста

        Вывод:
            Direction: Упростить, если текст сложнее ступени цели, усложнить, если проще,
                иначе оставить на уровне
        """
        gap = self.gap(grade)
        if gap > 0:
            return Direction.SIMPLIFY
        if gap < 0:
            return Direction.COMPLICATE
        return Direction.KEEP
