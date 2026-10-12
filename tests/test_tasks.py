import json

import pytest
from anthropic import transform_schema

from ruts_adapt.measure import Target, measure
from ruts_adapt.model import ModelError, Reply
from ruts_adapt.settings import ModelSettings
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
from tests.fakes import FakeClient, message

SOURCE = (
    "Настоящее постановление устанавливает порядок предоставления субсидий юридическим лицам "
    "в целях возмещения недополученных доходов в связи с осуществлением деятельности."
)
FACTS = [
    "Постановление устанавливает порядок субсидий",
    "Субсидии возмещают недополученные доходы",
]
SETTINGS = ModelSettings()


def checks(*statuses):
    return [
        FactCheck(fact=number, status=status, quote="" if status is FactStatus.LOST else "цитата")
        for number, status in statuses
    ]


def test_extract_facts():
    client = FakeClient(message(Facts(facts=[" Кот спит ", "", "Кот рыжий"])))
    reply = extract_facts(client, SETTINGS, "Рыжий кот спит.")
    assert reply == Reply(["Кот спит", "Кот рыжий"], "claude-opus-5-5", 120, 45)
    (request,) = client.requests
    assert request["system"] == FACTS_SYSTEM
    assert request["messages"][0]["content"] == "<текст>\nРыжий кот спит.\n</текст>"
    assert request["output_config"]["format"]["schema"] == transform_schema(Facts)


def test_extract_no_facts():
    client = FakeClient(message(Facts(facts=[" "])))
    with pytest.raises(ModelError, match=r"^Модель не нашла в тексте ни одного факта$"):
        extract_facts(client, SETTINGS, "...")


def test_rewrite():
    client = FakeClient(message(Draft(text="  Есть правила о деньгах.  ")))
    reply = rewrite(client, SETTINGS, SOURCE, measure(SOURCE), Target.from_grade(5), FACTS)
    assert reply.output == "Есть правила о деньгах."
    (request,) = client.requests
    assert request["system"] == REWRITE_SYSTEM
    prompt = request["messages"][0]["content"]
    assert prompt.startswith(
        "Упрости текст для читателя уровня «4-6-й класс (9-11 лет)». "
        "Сейчас текст читается как «аспирантура (старше 22 лет)»: в среднем 18.0 слова "
        "в предложении, 3.61 слога и 8.33 буквы в слове, 50% слов от пяти слогов."
    )
    assert f"<исходный_текст>\n{SOURCE}\n</исходный_текст>" in prompt
    assert f"<факты>\n1. {FACTS[0]}\n2. {FACTS[1]}\n</факты>" in prompt
    assert "<прошлая_попытка>" not in prompt
    assert "<критика>" not in prompt


def test_rewrite_retry():
    """Новая попытка переписывает исходник, видя прошлый черновик и критику к нему"""
    client = FakeClient(message(Draft(text="Текст")))
    rewrite(
        client,
        SETTINGS,
        SOURCE,
        measure(SOURCE),
        Target.from_grade(5),
        FACTS,
        previous="Прошлый черновик",
        critique="Предложения длинные",
    )
    prompt = client.requests[0]["messages"][0]["content"]
    assert prompt.endswith(
        "<прошлая_попытка>\nПрошлый черновик\n</прошлая_попытка>\n\n<критика>\nПредложения длинные\n</критика>"
    )
    assert f"<исходный_текст>\n{SOURCE}\n</исходный_текст>" in prompt


def test_rewrite_complicate():
    text = "Кот спит. Кот ест."
    client = FakeClient(message(Draft(text="Текст")))
    rewrite(client, SETTINGS, text, measure(text), Target.from_grade(10), ["Кот спит"])
    assert client.requests[0]["messages"][0]["content"].startswith(
        "Усложни текст для читателя уровня «10-11-й класс (15-16 лет)»"
    )


def test_rewrite_on_level():
    text = "Кот спит. Кот ест."
    with pytest.raises(ValueError, match=r"^Текст уже на уровне цели: 1-3-й класс"):
        rewrite(FakeClient(), SETTINGS, text, measure(text), Target.from_grade(2), ["Кот спит"])


def test_rewrite_empty():
    client = FakeClient(message(Draft(text=" \n")))
    with pytest.raises(ModelError, match=r"^Модель вернула пустой текст$"):
        rewrite(client, SETTINGS, SOURCE, measure(SOURCE), Target.from_grade(5), FACTS)


def test_judge():
    verdict = Verdict(checks=checks((2, FactStatus.LOST), (1, FactStatus.KEPT)))
    client = FakeClient(message(verdict))
    reply = judge(client, SETTINGS, FACTS, "Есть правила о деньгах.")
    assert reply.output == checks((1, FactStatus.KEPT), (2, FactStatus.LOST))
    (request,) = client.requests
    assert request["system"] == JUDGE_SYSTEM
    assert request["messages"][0]["content"] == (
        f"<факты>\n1. {FACTS[0]}\n2. {FACTS[1]}\n</факты>\n\n"
        "<пересказ>\nЕсть правила о деньгах.\n</пересказ>"
    )


def test_verdict_schema():
    """Модель видит статусы по-русски, как в промпте судьи"""
    schema = json.dumps(transform_schema(Verdict), ensure_ascii=False)
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
    client = FakeClient(message(Verdict(checks=checks(*statuses))))
    with pytest.raises(ModelError) as info:
        judge(client, SETTINGS, FACTS, "Черновик")
    assert str(info.value) == f"Судья оценил не каждый факт ровно один раз ({problems})"
