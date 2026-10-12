from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

USAGE = {"input_tokens": 120, "output_tokens": 45, "total_tokens": 165}


def _text(output: BaseModel | str) -> str:
    return output if isinstance(output, str) else output.model_dump_json()


def anthropic(
    output: BaseModel | str = "",
    stop_reason: str = "end_turn",
    category: str | None = None,
    partial: str | None = None,
) -> AIMessage:
    """
    Ответ Claude, как его собирает ChatAnthropic из потока: блок мышления и текст

    С partial ответ повторяет отказ основной модели посреди потока: ее частичный
    текст и ответ резервной модели отдельными блоками; блок fallback LangChain
    из потока выбрасывает, а имя модели берет из начала потока - основной
    """
    content: list[str | dict[str, Any]] = [{"type": "thinking", "thinking": "", "index": 0}]
    if partial is not None:
        content.append({"type": "text", "text": partial, "index": 1})
    content.append({"type": "text", "text": _text(output), "index": 3})
    metadata: dict[str, Any] = {
        "model_name": "claude-opus-5-5",
        "model_provider": "anthropic",
        "stop_reason": stop_reason,
        "stop_sequence": None,
    }
    if stop_reason == "refusal":
        metadata["stop_details"] = {"type": "refusal", "category": category, "explanation": None}
    return AIMessage(content=content, response_metadata=metadata, usage_metadata=USAGE)


def openai(
    output: BaseModel | str = "",
    finish_reason: str | None = "stop",
    refusal: str | None = None,
    **metadata: Any,
) -> AIMessage:
    """Ответ OpenAI, как его собирает ChatOpenAI; metadata - поля Responses API"""
    return AIMessage(
        content=_text(output),
        additional_kwargs={"refusal": refusal},
        response_metadata={
            "model_name": "gpt-5-2025-08-07",
            "model_provider": "openai",
            "finish_reason": finish_reason,
            **metadata,
        },
        usage_metadata=USAGE,
    )


def gemini(
    output: BaseModel | str = "", finish_reason: str = "STOP", blocked: Any = None
) -> AIMessage:
    """Ответ Gemini, как его собирает ChatGoogleGenerativeAI; заблокированный - без кандидатов"""
    if blocked is not None:
        feedback = {"block_reason": blocked, "block_reason_message": None, "safety_ratings": None}
        return AIMessage(content="", response_metadata={"prompt_feedback": feedback})
    metadata = {
        "finish_reason": finish_reason,
        "model_name": "gemini-2.5-pro",
        "model_provider": "google_genai",
    }
    return AIMessage(content=_text(output), response_metadata=metadata, usage_metadata=USAGE)


def ollama(output: BaseModel | str = "", done_reason: str = "stop") -> AIMessage:
    """Ответ локальной модели, как его собирает ChatOllama из потока"""
    metadata = {
        "model": "qwen3:30b",
        "model_name": "qwen3:30b",
        "model_provider": "ollama",
        "done": True,
        "done_reason": done_reason,
    }
    return AIMessage(content=_text(output), response_metadata=metadata, usage_metadata=USAGE)


class RepliesExhaustedError(BaseException):
    """Ответы поддельной модели кончились: ошибка теста, а не ответ модели, поэтому ask ее не ловит"""


class FakeChatModel(BaseChatModel):
    """
    Поддельная модель чата: отдает заданные ответы по очереди и запоминает запросы

    Ответ-исключение бросается, как бросил бы его SDK провайдера, ответ-функция
    вызывается в момент запроса. Структурированный
    вывод отдает сырой ответ, как with_structured_output(include_raw=True) моделей
    LangChain; разбор по схеме делает проверяемый код
    """

    replies: list[Any]
    model: str = "fake"
    requests: list[list[BaseMessage]] = Field(default_factory=list)
    schemas: list[type[BaseModel]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        if not self.replies:
            raise RepliesExhaustedError("Поддельной модели не задан ответ на запрос")
        self.requests.append(messages)
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply()
        if isinstance(reply, BaseException):
            raise reply
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def with_structured_output(  # type: ignore[override]
        self,
        schema: type[BaseModel],
        *,
        include_raw: bool = False,
        method: str = "",
        **kwargs: Any,
    ) -> Runnable[Any, dict[str, Any]]:
        assert (method, include_raw) == ("json_schema", True), (method, include_raw)
        self.schemas.append(schema)
        return self | RunnableLambda(
            lambda raw: {"raw": raw, "parsed": None, "parsing_error": None}
        )
