import pytest

from ruts_adapt.language import check_russian


@pytest.mark.parametrize(
    "text",
    ["Мама мыла раму", "Чтó это? Ёлка!", "В 2026 году ruTS считает слоги: Python - язык."],
    ids=["plain", "marks", "mixed"],
)
def test_russian(text):
    check_russian(text)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "В тексте нет букв: адаптировать нечего"),
        ("123 - 456!", "В тексте нет букв: адаптировать нечего"),
        (
            "Mama washed the frame",
            "Букв русского алфавита - только 0%: ruts-adapt адаптирует только русские тексты",
        ),
        (
            "Кот cats",
            "Букв русского алфавита - только 42%: ruts-adapt адаптирует только русские тексты",
        ),
        (
            "Він прийшов додому",
            "Букв кириллицы не из русского алфавита - 6% (і): текст, похоже, не на русском "
            "языке, а ruts-adapt адаптирует только русские тексты",
        ),
    ],
    ids=["empty", "no-letters", "english", "half", "ukrainian"],
)
def test_not_russian(text, message):
    with pytest.raises(ValueError) as info:
        check_russian(text)
    assert str(info.value) == message
