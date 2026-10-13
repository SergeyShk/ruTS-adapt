import re

from ruts.utils import strip_marks

MIN_RUSSIAN_SHARE = 0.5
MAX_FOREIGN_CYRILLIC_SHARE = 0.03
RUSSIAN = re.compile(r"[а-яё]", re.IGNORECASE)
FOREIGN_CYRILLIC = re.compile(r"(?![а-яё])[\u0400-\u052f]", re.IGNORECASE)


def russian_share(text: str) -> float:
    """
    Доля букв русского алфавита среди всех букв текста

    Описание:
        Буквы считаются без знаков ударения (strip_marks); в тексте без букв - 0

    Аргументы:
        text (str): Текст

    Вывод:
        float: Доля от 0 до 1

    Пример использования:
        >>> russian_share("Дуб (Quercus robur)")
        0.2
    """
    text = strip_marks(text)
    letters = sum(char.isalpha() for char in text)
    return len(RUSSIAN.findall(text)) / letters if letters else 0.0


def check_russian(text: str, min_share: float = MIN_RUSSIAN_SHARE) -> None:
    """
    Проверка, что текст на русском языке

    Описание:
        ruTS считает слоги и слова по правилам русского языка, поэтому уровень
        текста на другом языке или без букв не имеет смысла. Ошибку дают текст
        без букв, доля букв русского алфавита ниже min_share и от 3 % букв кириллицы
        не из него (і, ї, ў, ј, қ), как в украинском, белорусском, сербском или
        казахском тексте; доля в сообщении округляется вниз. Буквы считаются
        без знаков ударения (strip_marks): ѐ, ѝ и «чтó» - русские

    Аргументы:
        text (str): Текст
        min_share (float): Наименьшая доля букв русского алфавита

    Исключения:
        ValueError: Если в тексте нет букв или он, похоже, не на русском языке

    Пример использования:
        >>> check_russian("Мама мыла раму")
        >>> check_russian("Mama washed the frame")
        Traceback (most recent call last):
        ...
        ValueError: Букв русского алфавита - только 0%: ruts-adapt адаптирует только русские тексты
    """
    text = strip_marks(text)
    letters = sum(char.isalpha() for char in text)
    if not letters:
        raise ValueError("В тексте нет букв: адаптировать нечего")
    russian = len(RUSSIAN.findall(text))
    if russian / letters < min_share:
        raise ValueError(
            f"Букв русского алфавита - только {_percent(russian, letters)}: "
            "ruts-adapt адаптирует только русские тексты"
        )
    foreign = FOREIGN_CYRILLIC.findall(text)
    if len(foreign) / letters >= MAX_FOREIGN_CYRILLIC_SHARE:
        examples = ", ".join(sorted({char.lower() for char in foreign}))
        raise ValueError(
            f"Букв кириллицы не из русского алфавита - {_percent(len(foreign), letters)} "
            f"({examples}): текст, похоже, не на русском языке, а ruts-adapt адаптирует "
            "только русские тексты"
        )


def _percent(count: int, total: int) -> str:
    """Доля в процентах с округлением вниз, чтобы 49,9 % не выглядели как пороговые 50 %"""
    return f"{100 * count // total}%"
