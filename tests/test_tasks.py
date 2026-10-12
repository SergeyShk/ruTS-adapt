import json

import pytest

from ruts_adapt.measure import Target, measure
from ruts_adapt.model import ModelError, Reply
from ruts_adapt.tasks import (
    FACTS_SYSTEM,
    JUDGE_SYSTEM,
    REWRITE_SYSTEM,
    Draft,
    FactCheck,
    Facts,
    FactStatus,
    Verdict,
    extract_facts,
    judge,
    rewrite,
)
from tests.fakes import FakeChatModel, anthropic

SOURCE = (
    "Настоящее постановление устанавливает порядок предоставления субсидий юридическим лицам "
    "в целях возмещения недополученных доходов в связи с осуществлением деятельности."
)
FACTS = [
    "Постановление устанавливает порядок субсидий",
    "Субсидии возмещают недополученные доходы",
]


def fake(*outputs):
    return FakeChatModel(replies=[anthropic(output) for output in outputs])


def sent(model):
    """Системный промпт и сообщение единственного запроса"""
    ((system, human),) = model.requests
    return system.content, human.content


def checks(*statuses):
    return [
        FactCheck(fact=number, status=status, quote="" if status is FactStatus.LOST else "цитата")
        for number, status in statuses
    ]


def test_extract_facts():
    model = fake(Facts(facts=[" Кот спит ", "", "Кот рыжий"]))
    reply = extract_facts(model, "Рыжий кот спит.")
    assert reply == Reply(["Кот спит", "Кот рыжий"], "claude-opus-5-5", 120, 45)
    assert sent(model) == (FACTS_SYSTEM, "<текст>\nРыжий кот спит.\n</текст>")
    assert model.schemas == [Facts]


def test_extract_no_facts():
    with pytest.raises(ModelError, match=r"^Модель не нашла в тексте ни одного факта$") as info:
        extract_facts(fake(Facts(facts=[" "])), "...")
    assert info.value.retryable


def test_rewrite():
    model = fake(Draft(text="  Есть правила о деньгах.  "))
    reply = rewrite(model, SOURCE, measure(SOURCE), Target.from_grade(5), FACTS)
    assert reply.output == "Есть правила о деньгах."
    system, prompt = sent(model)
    assert system == REWRITE_SYSTEM
    assert model.schemas == [Draft]
    assert prompt.startswith(
        "Упрости исходный текст для читателя уровня «4-6-й класс (9-11 лет)». "
        "Сейчас он читается как «аспирантура (старше 22 лет)»: в среднем 18.0 слова "
        "в предложении, 3.61 слога и 8.33 буквы в слове, 50% слов от пяти слогов."
    )
    assert f"<исходный_текст>\n{SOURCE}\n</исходный_текст>" in prompt
    assert f"<факты>\n1. {FACTS[0]}\n2. {FACTS[1]}\n</факты>" in prompt
    assert "<прошлая_попытка>" not in prompt
    assert "<критика>" not in prompt


def test_rewrite_retry():
    """Новая попытка переписывает исходник, видя прошлый черновик и критику к нему"""
    model = fake(Draft(text="Текст"))
    rewrite(
        model,
        SOURCE,
        measure(SOURCE),
        Target.from_grade(5),
        FACTS,
        previous="Прошлый черновик",
        critique="Предложения длинные",
    )
    system, prompt = sent(model)
    assert "Перепиши заново исходный текст, учитывая критику" in system
    assert prompt.endswith(
        "<прошлая_попытка>\nПрошлый черновик\n</прошлая_попытка>\n\n<критика>\nПредложения длинные\n</критика>"
    )
    assert f"<исходный_текст>\n{SOURCE}\n</исходный_текст>" in prompt


def test_rewrite_complicate():
    text = "Кот спит. Кот ест."
    model = fake(Draft(text="Текст"))
    rewrite(model, text, measure(text), Target.from_grade(10), ["Кот спит"])
    assert sent(model)[1].startswith(
        "Усложни исходный текст для читателя уровня «10-11-й класс (15-16 лет)»"
    )


def test_rewrite_on_level():
    text = "Кот спит. Кот ест."
    with pytest.raises(ValueError, match=r"^Текст уже на уровне цели: 1-3-й класс"):
        rewrite(fake(), text, measure(text), Target.from_grade(2), ["Кот спит"])


def test_rewrite_empty():
    with pytest.raises(ModelError, match=r"^Модель вернула пустой текст$") as info:
        rewrite(fake(Draft(text=" \n")), SOURCE, measure(SOURCE), Target.from_grade(5), FACTS)
    assert info.value.retryable


@pytest.mark.parametrize(
    ("draft", "message"),
    [
        ("Rules about money.", "Букв русского алфавита - только 0%"),
        ("…", "В тексте нет букв"),
    ],
    ids=["english", "no-letters"],
)
def test_rewrite_not_russian(draft, message):
    """Черновик проверяется, как вход: не русский текст - ошибка, которую стоит повторить"""
    with pytest.raises(ModelError, match=f"^Черновик модели не подходит: {message}") as info:
        rewrite(fake(Draft(text=draft)), SOURCE, measure(SOURCE), Target.from_grade(5), FACTS)
    assert info.value.retryable


def test_judge():
    verdict = Verdict(checks=checks((2, FactStatus.LOST), (1, FactStatus.KEPT)))
    model = fake(verdict)
    reply = judge(model, FACTS, "Есть правила о деньгах.")
    assert reply.output == checks((1, FactStatus.KEPT), (2, FactStatus.LOST))
    assert model.schemas == [Verdict]
    assert sent(model) == (
        JUDGE_SYSTEM,
        f"<факты>\n1. {FACTS[0]}\n2. {FACTS[1]}\n</факты>\n\n"
        "<пересказ>\nЕсть правила о деньгах.\n</пересказ>",
    )


def test_verdict_schema():
    """Модель видит статусы по-русски, как в промпте судьи"""
    schema = json.dumps(Verdict.model_json_schema(), ensure_ascii=False)
    assert '["сохранен", "искажен", "потерян"]' in schema
    for status in FactStatus:
        assert f"- {status} - " in JUDGE_SYSTEM


@pytest.mark.parametrize(
    ("statuses", "problems"),
    [
        ([(1, FactStatus.KEPT)], "нет оценки: 2"),
        (
            [(1, FactStatus.KEPT), (1, FactStatus.LOST), (2, FactStatus.KEPT)],
            "несколько оценок: 1",
        ),
        (
            [(1, FactStatus.KEPT), (3, FactStatus.KEPT), (0, FactStatus.LOST)],
            "нет оценки: 2; нет такого факта: 0, 3",
        ),
    ],
)
def test_judge_incomplete(statuses, problems):
    with pytest.raises(ModelError) as info:
        judge(fake(Verdict(checks=checks(*statuses))), FACTS, "Черновик")
    assert str(info.value) == f"Судья оценил не каждый факт ровно один раз ({problems})"
    assert info.value.retryable
