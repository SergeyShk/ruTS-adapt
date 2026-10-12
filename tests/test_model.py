import pytest
from anthropic import transform_schema
from pydantic import BaseModel, ValidationError

from ruts_adapt.model import FALLBACKS_BETA, ModelError, Reply, ask
from ruts_adapt.settings import ModelSettings
from tests.fakes import FakeClient, message


class Answer(BaseModel):
    words: list[str]


ANSWER = Answer(words=["кот", "спит"])


def test_ask():
    client = FakeClient(message(ANSWER))
    reply = ask(client, ModelSettings(), "Системный промпт", "Вопрос", Answer)
    assert reply == Reply(ANSWER, "claude-opus-5-5", 120, 45)
    (request,) = client.requests
    assert request == {
        "model": "claude-opus-5-5",
        "max_tokens": 64_000,
        "system": "Системный промпт",
        "messages": [{"role": "user", "content": "Вопрос"}],
        "output_config": {
            "effort": "high",
            "format": {"type": "json_schema", "schema": transform_schema(Answer)},
        },
        "fallbacks": "default",
        "betas": [FALLBACKS_BETA],
    }


def test_ask_settings():
    """Модель, усилие и лимит берутся из настроек; без резервной модели нет и беты"""
    client = FakeClient(message(ANSWER, model="claude-sonnet-5-5"))
    settings = ModelSettings("claude-sonnet-5-5", "low", 1000, fallbacks=False)
    reply = ask(client, settings, "", "Вопрос", Answer)
    assert reply.model == "claude-sonnet-5-5"
    (request,) = client.requests
    assert (request["model"], request["max_tokens"]) == ("claude-sonnet-5-5", 1000)
    assert request["output_config"]["effort"] == "low"
    assert "fallbacks" not in request
    assert "betas" not in request


@pytest.mark.parametrize("partial", [None, '{"words": ["ко'])
def test_ask_fallback_model(partial):
    """Ответ резервной модели разбирается без частичного текста основной, видно, кто отвечал"""
    client = FakeClient(message(ANSWER, model="claude-opus-5", partial=partial))
    reply = ask(client, ModelSettings(), "", "Вопрос", Answer)
    assert reply == Reply(ANSWER, "claude-opus-5", 120, 45)


@pytest.mark.parametrize(
    ("category", "text"),
    [("bio", "категория отказа: bio"), (None, "категория отказа: не указана")],
)
def test_refusal(category, text):
    client = FakeClient(message("", stop_reason="refusal", category=category))
    with pytest.raises(
        ModelError, match=f"^Модель claude-opus-5-5 отказалась отвечать \\({text}\\)$"
    ):
        ask(client, ModelSettings(), "", "Вопрос", Answer)


def test_max_tokens():
    """Обрезанный JSON не разбирается: ошибка называет лимит, а не схему"""
    client = FakeClient(message('{"words": ["кот", "сп', stop_reason="max_tokens"))
    with pytest.raises(ModelError, match=r"^Ответ модели обрезан на max_tokens=500: увеличьте"):
        ask(client, ModelSettings(max_tokens=500), "", "Вопрос", Answer)


def test_unexpected_stop():
    client = FakeClient(message(ANSWER, stop_reason="pause_turn"))
    with pytest.raises(ModelError, match=r"неожиданной причине: pause_turn$"):
        ask(client, ModelSettings(), "", "Вопрос", Answer)


@pytest.mark.parametrize("text", ['{"words": "кот"}', "кот спит", ""])
def test_schema_mismatch(text):
    client = FakeClient(message(text))
    with pytest.raises(ModelError, match=r"^Ответ модели не соответствует схеме Answer$") as info:
        ask(client, ModelSettings(), "", "Вопрос", Answer)
    assert isinstance(info.value.__cause__, ValidationError)


def test_with_output():
    reply = Reply(ANSWER, "claude-opus-5-5", 120, 45)
    assert reply.with_output(ANSWER.words) == Reply(["кот", "спит"], "claude-opus-5-5", 120, 45)
