from types import SimpleNamespace

import pytest
from google.genai.types import BlockedReason
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from openai import ContentFilterFinishReasonError, LengthFinishReasonError
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ValidationError

from ruts_adapt.model import FALLBACKS_BETA, ModelError, Reply, ask, chat_model
from ruts_adapt.settings import ModelSettings
from tests.fakes import FakeChatModel, anthropic, gemini, ollama, openai


class Answer(BaseModel):
    words: list[str]


ANSWER = Answer(words=["кот", "спит"])


@pytest.fixture
def keys(monkeypatch):
    """Ключи-заглушки: модели LangChain требуют ключ при создании, запросов тесты не шлют"""
    for variable in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(variable, "test")


def test_chat_model_anthropic(keys):
    model = chat_model(ModelSettings())
    assert isinstance(model, ChatAnthropic)
    assert (model.model, model.max_tokens, model.reasoning_effort) == (
        "claude-opus-5-5",
        64_000,
        "high",
    )
    assert model.streaming is True
    assert model.betas == [FALLBACKS_BETA]
    assert model.model_kwargs == {"fallbacks": "default"}


def test_chat_model_anthropic_options(keys):
    """Без резервной модели нет ни беты, ни параметра; пустое усилие - умолчание провайдера"""
    settings = ModelSettings("anthropic:claude-sonnet-5-5", None, 16_000, fallbacks=False)
    model = chat_model(settings)
    assert (model.model, model.max_tokens, model.reasoning_effort) == (
        "claude-sonnet-5-5",
        16_000,
        None,
    )
    assert model.betas is None
    assert model.model_kwargs == {}


def test_chat_model_openai(keys):
    """Резервная модель - только у Claude, поток не нужен"""
    model = chat_model(ModelSettings("openai:gpt-5", "medium"))
    assert isinstance(model, ChatOpenAI)
    assert (model.model_name, model.max_tokens, model.reasoning_effort) == (
        "gpt-5",
        64_000,
        "medium",
    )
    assert model.streaming is False
    assert model.model_kwargs == {}


def test_chat_model_provider_defaults(keys):
    """Без усилия и лимита модель получает умолчания провайдера"""
    model = chat_model(ModelSettings("openai:gpt-4.1", None, None))
    assert (model.reasoning_effort, model.max_tokens) == (None, None)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (
            ModelSettings("openai:gpt-5", "xhigh"),
            "Модель openai:gpt-5 не поддерживает усилие 'xhigh': допустимо minimal, low, medium, high",
        ),
        (
            ModelSettings("openai:gpt-4.1", "high", None),
            "Модель openai:gpt-4.1 не поддерживает усилие 'high': оставьте усилие пустым",
        ),
        (
            ModelSettings("openai:gpt-4.1", None, 64_000),
            "max_tokens=64000 больше предела модели openai:gpt-4.1: 32768",
        ),
    ],
    ids=["effort-level", "no-reasoning", "max-tokens"],
)
def test_chat_model_profile(keys, settings, message):
    """Усилие и лимит сверяются с профилем модели LangChain"""
    with pytest.raises(ValueError) as info:
        chat_model(settings)
    assert str(info.value) == message


def test_chat_model_gemini(keys):
    model = chat_model(ModelSettings("google_genai:gemini-2.5-pro", "low", 8000))
    assert isinstance(model, ChatGoogleGenerativeAI)
    assert (model.model, model.max_output_tokens, model.reasoning_effort) == (
        "gemini-2.5-pro",
        8000,
        "low",
    )
    assert not model.streaming


@pytest.mark.parametrize(
    ("effort", "max_tokens", "reasoning"),
    [("high", 8000, "high"), ("none", 8000, False), (None, None, None)],
)
def test_chat_model_ollama(effort, max_tokens, reasoning):
    """Локальной модели ключ не нужен; лимит - num_predict, none выключает рассуждение"""
    model = chat_model(ModelSettings("ollama:qwen3:30b", effort, max_tokens))
    assert isinstance(model, ChatOllama)
    assert (model.model, model.num_predict, model.reasoning) == (
        "qwen3:30b",
        max_tokens,
        reasoning,
    )


@pytest.mark.parametrize(
    ("reply", "name"),
    [
        (anthropic(ANSWER), "claude-opus-5-5"),
        (openai(ANSWER), "gpt-5-2025-08-07"),
        (openai(ANSWER, finish_reason=None, status="completed"), "gpt-5-2025-08-07"),
        (gemini(ANSWER), "gemini-2.5-pro"),
        (ollama(ANSWER), "qwen3:30b"),
    ],
    ids=["anthropic", "openai", "openai-responses", "gemini", "ollama"],
)
def test_ask(reply, name):
    model = FakeChatModel(replies=[reply])
    assert ask(model, "Системный промпт", "Вопрос", Answer) == Reply(ANSWER, name, 120, 45)
    assert model.requests == [[SystemMessage("Системный промпт"), HumanMessage("Вопрос")]]
    assert model.schemas == [Answer]


@pytest.mark.parametrize(
    ("partial", "text"),
    [('{"words": ["ко', ANSWER.model_dump_json()), ('{"words": ["', 'кот", "спит"]}')],
    ids=["restart", "continue"],
)
def test_ask_fallback(partial, text):
    """После отказа Claude резервная модель начинает заново или продолжает частичный текст"""
    reply = anthropic(text, partial=partial)
    result = ask(FakeChatModel(replies=[reply]), "", "Вопрос", Answer)
    assert result == Reply(ANSWER, "claude-opus-5-5", 120, 45)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            "Ответ модели fake обрезан по лимиту токенов: увеличьте MAX_TOKENS в настройках",
        ),
        (
            {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}},
            "Модель fake отказалась отвечать (причина: content_filter)",
        ),
        ({"status": "completed"}, "Ответ модели не соответствует схеме Answer"),
        (None, "Ответ модели не соответствует схеме Answer"),
        (ValueError, "Ответ модели не соответствует схеме Answer"),
    ],
    ids=["max-output-tokens", "content-filter", "completed", "no-response", "not-json"],
)
def test_ask_responses_api_error(body, message):
    """В Responses API SDK OpenAI разбирает ответ сам; причину LangChain прикрепляет к ошибке"""
    try:
        Answer.model_validate_json('{"words": ["ко')
    except ValidationError as error:
        failure = error

    def json():
        if body is ValueError:
            raise ValueError("не JSON")
        return body

    if body is not None:
        failure.response = SimpleNamespace(json=json)
    with pytest.raises(ModelError) as info:
        ask(FakeChatModel(replies=[failure]), "", "Вопрос", Answer)
    assert str(info.value) == message
    assert info.value.__cause__ is failure


def test_ask_fallback_merged():
    """Без мышления LangChain склеивает частичный текст и новый ответ резервной модели в строку"""
    reply = anthropic(ANSWER)
    reply.content = '{"words": ["ко{"words": ["кот", "спит"]}'
    assert ask(FakeChatModel(replies=[reply]), "", "Вопрос", Answer).output == ANSWER


def test_ask_think_tags():
    """Локальная модель с рассуждением по умолчанию пишет его перед ответом в тегах think"""
    reply = ollama('<think>Нужен объект {"words": [...]}</think>' + ANSWER.model_dump_json())
    assert ask(FakeChatModel(replies=[reply]), "", "Вопрос", Answer).output == ANSWER


def test_ask_string_blocks():
    """Блоки содержимого LangChain бывают и строками"""
    reply = anthropic(ANSWER)
    reply.content = ['{"words": ["кот", ', '"спит"]}']
    assert ask(FakeChatModel(replies=[reply]), "", "Вопрос", Answer).output == ANSWER


@pytest.mark.parametrize(
    ("reply", "detail", "name"),
    [
        (anthropic(stop_reason="refusal", category="bio"), "bio", "claude-opus-5-5"),
        (anthropic(stop_reason="refusal"), "не указана", "claude-opus-5-5"),
        (openai(refusal="I can't help with that."), "I can't help with that.", "gpt-5-2025-08-07"),
        (openai(finish_reason="content_filter"), "content_filter", "gpt-5-2025-08-07"),
        (
            openai(
                finish_reason=None,
                status="incomplete",
                incomplete_details={"reason": "content_filter"},
            ),
            "content_filter",
            "gpt-5-2025-08-07",
        ),
        (
            AIMessage(
                content=[{"type": "refusal", "refusal": "Нет"}],
                response_metadata={
                    "model_provider": "openai",
                    "model_name": "gpt-5",
                    "status": "completed",
                },
            ),
            "Нет",
            "gpt-5",
        ),
        (gemini(finish_reason="SAFETY"), "SAFETY", "gemini-2.5-pro"),
        (gemini(blocked=BlockedReason.SAFETY), "SAFETY", "fake"),
        (gemini(blocked="PROHIBITED_CONTENT"), "PROHIBITED_CONTENT", "fake"),
        (ContentFilterFinishReasonError(), "content_filter", "fake"),
    ],
    ids=[
        "anthropic",
        "anthropic-no-category",
        "openai",
        "openai-filter",
        "openai-responses-filter",
        "openai-responses-refusal",
        "gemini",
        "gemini-blocked",
        "gemini-blocked-string",
        "openai-sdk-filter",
    ],
)
def test_refusal(reply, detail, name):
    """Отказ - ошибка с причиной; имя модели без метаданных берется из ее настроек"""
    with pytest.raises(ModelError) as info:
        ask(FakeChatModel(replies=[reply]), "", "Вопрос", Answer)
    assert str(info.value) == f"Модель {name} отказалась отвечать (причина: {detail})"


@pytest.mark.parametrize(
    ("reply", "name"),
    [
        (anthropic('{"words": ["ко', stop_reason="max_tokens"), "claude-opus-5-5"),
        (openai('{"words": ["ко', finish_reason="length"), "gpt-5-2025-08-07"),
        (
            openai(
                finish_reason=None,
                status="incomplete",
                incomplete_details={"reason": "max_output_tokens"},
            ),
            "gpt-5-2025-08-07",
        ),
        (gemini('{"words": ["ко', finish_reason="MAX_TOKENS"), "gemini-2.5-pro"),
        (LengthFinishReasonError(completion=ChatCompletion.model_construct(usage=None)), "fake"),
        (ollama('{"words": ["ко', done_reason="length"), "qwen3:30b"),
    ],
    ids=["anthropic", "openai", "openai-responses", "gemini", "openai-sdk", "ollama"],
)
def test_truncated(reply, name):
    """Обрезанный JSON не разбирается: ошибка называет лимит, а не схему"""
    with pytest.raises(ModelError) as info:
        ask(FakeChatModel(replies=[reply]), "", "Вопрос", Answer)
    assert str(info.value) == (
        f"Ответ модели {name} обрезан по лимиту токенов: увеличьте MAX_TOKENS в настройках"
    )


@pytest.mark.parametrize(
    ("reply", "detail"),
    [
        (anthropic(ANSWER, stop_reason="pause_turn"), "pause_turn"),
        (openai(ANSWER, finish_reason="tool_calls"), "tool_calls"),
        (gemini(ANSWER, finish_reason="OTHER"), "OTHER"),
        (gemini(blocked="BLOCKED_REASON_UNSPECIFIED"), "None"),
        (ollama(ANSWER, done_reason="load"), "load"),
        (AIMessage(content=ANSWER.model_dump_json()), "неизвестный провайдер None"),
    ],
    ids=["anthropic", "openai", "gemini", "gemini-unspecified", "ollama", "unknown"],
)
def test_unexpected_stop(reply, detail):
    with pytest.raises(ModelError, match=f"остановилась по неожиданной причине: {detail}$"):
        ask(FakeChatModel(replies=[reply]), "", "Вопрос", Answer)


@pytest.mark.parametrize(
    "reply",
    [anthropic('{"words": "кот"}'), openai("кот спит"), gemini(""), anthropic("{}", partial="{")],
    ids=["wrong-type", "not-json", "empty", "blocks"],
)
def test_schema_mismatch(reply):
    with pytest.raises(ModelError, match=r"^Ответ модели не соответствует схеме Answer$") as info:
        ask(FakeChatModel(replies=[reply]), "", "Вопрос", Answer)
    assert isinstance(info.value.__cause__, ValidationError)


def test_ask_no_usage():
    """Провайдер не сообщил токены - в ответе нули"""
    reply = gemini(ANSWER)
    reply.usage_metadata = None
    reply = ask(FakeChatModel(replies=[reply]), "", "Вопрос", Answer)
    assert (reply.input_tokens, reply.output_tokens) == (0, 0)


def test_with_output():
    reply = Reply(ANSWER, "claude-opus-5-5", 120, 45)
    assert reply.with_output(ANSWER.words) == Reply(["кот", "спит"], "claude-opus-5-5", 120, 45)
