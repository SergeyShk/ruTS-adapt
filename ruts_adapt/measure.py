from dataclasses import dataclass
from enum import StrEnum

from ruts import BasicStats, ReadabilityStats
from ruts.constants import GRADE_AGE_LEVELS
from ruts.readability_stats import grade_to_age, level_features, level_norms

MIN_GRADE = GRADE_AGE_LEVELS[0][0]
MAX_GRADE = GRADE_AGE_LEVELS[-1][1]
# На тексте короче сводный класс неустойчив: слияние двух предложений нередко меняет ступень
STABLE_WORDS = 100
# Нормы ruTS для 10-11-го класса граф не берет: тексты в них сводный класс относит к 4-6 и 7-9
NORM_LEVELS = (1, 4)


def number(value: float, digits: int = 1) -> str:
    """
    Число с десятичной запятой для текстов по-русски

    Аргументы:
        value (float): Число
        digits (int): Знаков после запятой

    Вывод:
        str: Число с запятой

    Пример использования:
        >>> number(4.65), number(12.0, 0), number(3.125, 2)
        ('4,7', '12', '3,12')
    """
    return f"{value:.{digits}f}".replace(".", ",")


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
        и его прочтение как ступени обучения и возраста читателя. Признаки -
        величины, из которых складываются эти формулы: длина предложения
        (все формулы), длина слова в слогах (Флеш-Кинкайд, индекс Флеша)
        и в буквах (Колман-Лиау, ARI, формула Соловьева, Иванова, Солнышкиной),
        доля сложных слов - от 5 слогов (SMOG, Дейл-Чейл, Ганнинг)

    Атрибуты:
        grade (float): Сводный класс
        level (str): Ступень обучения и возраст читателя по сводному классу
        n_words (int): Количество слов
        n_sents (int): Количество предложений
        words_per_sent (float): Средняя длина предложения в словах
        syllables_per_word (float): Средняя длина слова в слогах
        letters_per_word (float): Средняя длина слова в буквах
        complex_words (float): Доля сложных слов
        mtld (float): Лексическое разнообразие MTLD; inf, если в тексте нет
            повторяющихся слов
    """

    grade: float
    level: str
    n_words: int
    n_sents: int
    words_per_sent: float
    syllables_per_word: float
    letters_per_word: float
    complex_words: float
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
        >>> m.words_per_sent, m.syllables_per_word, m.letters_per_word, m.complex_words
        (9.0, 2.0, 5.0, 0.0)
    """
    bs = BasicStats(text)
    rs = ReadabilityStats(bs)
    features = level_features(text)
    return Measurement(
        grade=rs.consensus_grade,
        level=rs.describe_grade(),
        n_words=bs.n_words,
        n_sents=bs.n_sents,
        words_per_sent=features["words_per_sent"],
        syllables_per_word=bs.n_syllables / bs.n_words,
        letters_per_word=features["letters_per_word"],
        complex_words=features["p_smog_complex_words"],
        mtld=features["mtld"],
    )


@dataclass(frozen=True)
class Target:
    """
    Цель адаптации - ступень обучения читателя

    Описание:
        Класс читателя относится к своей ступени таблицы ruTS GRADE_AGE_LEVELS:
        1-3, 4-6, 7-9, 10-11-й класс, 1-3 и 4-6-й курс вуза (годы обучения 12-17).
        Текст на уровне, если ruTS прочитывает его сводный класс как эту ступень.
        Нормы признаков ступени - из ruTS (level_norms), только для 1-3 и 4-6-го класса

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
        >>> target.norm["words_per_sent"], Target.from_grade(8).norm, Target.from_grade(10).norm
        ((10.2, 14.6), None, None)
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

    @property
    def norm(self) -> dict[str, tuple[float, float]] | None:
        """Нормы признаков ступени цели по ключам level_features; None, если норм нет"""
        return level_norms(self.grade) if self.low in NORM_LEVELS else None

    def describe_norm(self) -> str | None:
        """
        Нормы ступени цели для промпта и критики

        Вывод:
            str | None: Обычные длина предложения и слова и наибольшая обычная доля
                слов от пяти слогов; None, если норм нет

        Пример использования:
            >>> Target.from_grade(2).describe_norm()
            'обычно 8,1-10,7 слова в предложении, 4,56-5,03 буквы в слове и не больше 2,2% слов от пяти слогов'
        """
        norm = self.norm
        if norm is None:
            return None
        words, letters = norm["words_per_sent"], norm["letters_per_word"]
        return (
            f"обычно {number(words[0])}-{number(words[1])} слова в предложении, "
            f"{number(letters[0], 2)}-{number(letters[1], 2)} буквы в слове и не больше "
            f"{number(100 * norm['p_smog_complex_words'][1])}% слов от пяти слогов"
        )

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
