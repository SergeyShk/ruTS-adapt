import os
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, cast

from langchain.chat_models import init_chat_model
from langchain_core.exceptions import ContextOverflowError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import ContentFilterFinishReasonError, LengthFinishReasonError
from pydantic import BaseModel, ValidationError

from .settings import DEFAULT_MAX_TOKENS, ModelSettings

FALLBACKS_BETA = "server-side-fallback-2026-07-01"
# Переменные ключа API провайдера: модель читает первую заданную
KEYS = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google_genai": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
}
# Gemini через Vertex AI берет учетные данные Google Cloud, а не ключ
VERTEX = "GOOGLE_GENAI_USE_VERTEXAI"
# Секунды ожидания ответа: модели LangChain отключают умолчание SDK (10 минут) и ждут без конца
TIMEOUT = 600
# Локальная модель на процессоре отвечает дольше облачной
OLLAMA_TIMEOUT = 1800
# Ошибки посреди потока Claude: приходят со статусом 200, и SDK их не повторяет
TRANSIENT_STREAM_ERRORS = {"overloaded_error", "api_error"}
GEMINI_REFUSALS = {
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_RECITATION",
}

T = TypeVar("T")
U = TypeVar("U")
SchemaT = TypeVar("SchemaT", bound=BaseModel)
Stop = Literal["done", "refused", "truncated", "overflow", "other"]
_REFUSED = "Модель {name} отказалась отвечать (причина: {detail})"
_MISMATCH = "Ответ модели не соответствует схеме {schema}"
_TRUNCATED = "Ответ модели {name} обрезан по лимиту токенов: увеличьте MAX_TOKENS в настройках"
_OVERFLOW = "Текст не помещается в окно контекста модели {name}: сократите его"


class ModelError(Exception):
    """
    Ответ модели, который нельзя использовать: отказ, обрезанный или не по схеме

    Атрибуты:
        retryable (bool): Повтор того же запроса может помочь - ответ не по схеме,
            неожиданная остановка, перегрузка или сбой сервера, сеть, таймаут;
            отказ, обрезка, переполнение окна контекста и неверный ключ повторятся снова
    """

    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class Reply(Generic[T]):
    """
    Ответ модели

    Атрибуты:
        output (Any): Разобранный ответ
        model (str): Модель по данным провайдера; если Claude отказался посреди ответа
            и ответ дописала резервная модель, LangChain называет основную
        input_tokens (int): Токены запроса
        output_tokens (int): Токены ответа вместе с мышлением
    """

    output: T
    model: str
    input_tokens: int
    output_tokens: int

    def with_output(self, output: U) -> "Reply[U]":
        """Тот же ответ с другим разобранным содержимым"""
        return Reply(output, self.model, self.input_tokens, self.output_tokens)


def chat_model(settings: ModelSettings) -> BaseChatModel:
    """
    Модель чата LangChain по настройкам роли

    Описание:
        Ответ Claude приходит потоком: SDK Anthropic не отправляет без потока
        запрос, который может идти дольше 10 минут, а большой max_tokens
        считается таким. Усилие и лимит сверяются с профилем модели LangChain,
        если он их знает: модель Claude или Gemini с профилем без уровней
        усилия усилия не принимает. Модели Claude без профиля и без MAX_TOKENS
        получают лимит модели по умолчанию, а не 4096 токенов LangChain. Ключ
        API модель берет из переменной окружения провайдера (KEYS), Gemini через
        Vertex AI (VERTEX) - учетные данные Google Cloud; Ollama ключа не требует,
        адрес ее сервера - OLLAMA_HOST, по умолчанию localhost:11434. Окно
        контекста локальной модели задается в каждом запросе и заменяет
        умолчание сервера OLLAMA_CONTEXT_LENGTH. Ответ ждется не дольше TIMEOUT
        секунд, у Ollama - OLLAMA_TIMEOUT

    Аргументы:
        settings (ModelSettings): Провайдер, модель, усилие, max_tokens и резервная модель

    Вывод:
        BaseChatModel: Модель чата

    Исключения:
        ValueError: Если не задан ключ API провайдера, модель не поддерживает
            ответ по схеме или усилие, max_tokens больше ее предела или окна контекста
    """
    keys = KEYS.get(settings.provider, ())
    vertex = settings.provider == "google_genai" and os.environ.get(VERTEX, "").lower() in (
        "1",
        "true",
    )
    if keys and not vertex and not any(os.environ.get(key) for key in keys):
        raise ValueError(
            f"Нет ключа API для модели {settings.model}: задайте переменную окружения "
            + " или ".join(keys)
        )
    options: dict[str, Any] = {}
    if settings.provider == "ollama":
        options["client_kwargs"] = {"timeout": OLLAMA_TIMEOUT}
        if settings.context is not None:
            options["num_ctx"] = settings.context
        if settings.max_tokens is not None:
            if settings.context is not None and settings.max_tokens >= settings.context:
                raise ValueError(
                    f"max_tokens={settings.max_tokens} не меньше окна контекста модели "
                    f"{settings.model}: {settings.context}, для запроса места не останется"
                )
            options["num_predict"] = settings.max_tokens
        if settings.effort is not None:
            options["reasoning"] = False if settings.effort == "none" else settings.effort
    else:
        options["timeout"] = TIMEOUT
        if settings.max_tokens is not None:
            options["max_tokens"] = settings.max_tokens
        if settings.effort is not None:
            options["reasoning_effort"] = settings.effort
    if settings.provider == "anthropic":
        options["streaming"] = True
        if settings.fallbacks:
            options |= {"betas": [FALLBACKS_BETA], "model_kwargs": {"fallbacks": "default"}}
    model: BaseChatModel = init_chat_model(
        settings.name, model_provider=settings.provider, **options
    )
    profile = model.profile or {}
    # Только OpenAI подменяет схему вызовом инструмента; у Claude профили LangChain
    # помечают structured_output = false и модели, которые схему принимают
    if settings.provider == "openai" and profile.get("structured_output") is False:
        raise ValueError(f"Модель {settings.model} не поддерживает ответ по JSON-схеме")
    levels = profile.get("reasoning_effort_levels")
    # У o-моделей OpenAI в профиле нет уровней, хотя усилие они принимают
    if settings.effort is not None and (
        (levels is not None and settings.effort not in levels)
        or (profile and levels is None and settings.provider != "openai")
        or profile.get("reasoning_output") is False
    ):
        allowed = f"допустимо {', '.join(levels)}" if levels else "оставьте усилие пустым"
        raise ValueError(
            f"Модель {settings.model} не поддерживает усилие {settings.effort!r}: {allowed}"
        )
    limit = profile.get("max_output_tokens")
    if settings.max_tokens is not None and limit is not None and settings.max_tokens > limit:
        raise ValueError(
            f"max_tokens={settings.max_tokens} больше предела модели {settings.model}: {limit}"
        )
    if settings.provider == "anthropic" and settings.max_tokens is None and limit is None:
        model = model.model_copy(update={"max_tokens": DEFAULT_MAX_TOKENS})
    return model


def ask(model: BaseChatModel, system: str, prompt: str, schema: type[SchemaT]) -> Reply[SchemaT]:
    """
    Запрос к модели с ответом по схеме

    Описание:
        Ответ ограничен JSON-схемой Pydantic-модели средствами провайдера
        (structured outputs). Причина остановки проверяется до разбора: отказ
        и обрезанный по лимиту токенов ответ - ошибка, а не пустой результат

    Аргументы:
        model (BaseChatModel): Модель чата
        system (str): Системный промпт
        prompt (str): Сообщение пользователя
        schema (type[BaseModel]): Схема ответа

    Вывод:
        Reply: Ответ, разобранный по схеме

    Исключения:
        ModelError: Если модель отказалась отвечать, ответ обрезан по лимиту
            токенов, текст не поместился в окно контекста, ответ остановился
            по другой причине, не соответствует схеме или вызов модели не удался
            (ключ, лимиты, сеть); временные сбои - с retryable
    """
    structured = model.with_structured_output(schema, method="json_schema", include_raw=True)
    name = _model_name(model)
    # SDK OpenAI сам разбирает ответ по схеме и до LangChain сообщает об обрезке и фильтре
    try:
        result = structured.invoke([SystemMessage(system), HumanMessage(prompt)])
    except LengthFinishReasonError as error:
        raise ModelError(_TRUNCATED.format(name=name)) from error
    except ContentFilterFinishReasonError as error:
        raise ModelError(_REFUSED.format(name=name, detail="content_filter")) from error
    except ValidationError as error:
        # В Responses API SDK OpenAI разбирает ответ по схеме, не глядя на status
        reason = _incomplete_reason(error)
        if reason == "max_output_tokens":
            raise ModelError(_TRUNCATED.format(name=name)) from error
        if reason == "content_filter":
            raise ModelError(_REFUSED.format(name=name, detail=reason)) from error
        raise ModelError(_MISMATCH.format(schema=schema.__name__), retryable=True) from error
    except ContextOverflowError as error:
        raise ModelError(_OVERFLOW.format(name=name)) from error
    except Exception as error:
        raise ModelError(
            f"Модель {name} не ответила: {type(error).__name__}: {error}",
            retryable=_transient(error),
        ) from error
    raw: AIMessage = cast(dict[str, Any], result)["raw"]
    name = raw.response_metadata.get("model_name") or name
    stop, detail = _stop(raw)
    if stop == "refused":
        raise ModelError(_REFUSED.format(name=name, detail=detail or "не указана"))
    if stop == "truncated":
        raise ModelError(_TRUNCATED.format(name=name))
    if stop == "overflow":
        raise ModelError(_OVERFLOW.format(name=name))
    if stop == "other":
        raise ModelError(
            f"Модель {name} остановилась по неожиданной причине: {detail}", retryable=True
        )
    usage = raw.usage_metadata or {"input_tokens": 0, "output_tokens": 0}
    return Reply(_parse(raw, schema), name, usage["input_tokens"], usage["output_tokens"])


def _stop(raw: AIMessage) -> tuple[Stop, str | None]:
    """Чем закончился ответ и почему, по метаданным провайдера"""
    metadata = raw.response_metadata
    provider = metadata.get("model_provider")
    if provider == "anthropic":
        return _anthropic_stop(metadata)
    if provider == "openai":
        return _openai_stop(raw)
    # Заблокированный запрос Gemini приходит без имени провайдера, только с prompt_feedback
    if provider == "google_genai" or "prompt_feedback" in metadata:
        return _gemini_stop(metadata)
    if provider == "ollama":
        return _ollama_stop(metadata)
    return "other", f"неизвестный провайдер {provider}"


def _anthropic_stop(metadata: dict[str, Any]) -> tuple[Stop, str | None]:
    """Итог ответа Claude по stop_reason"""
    reason = metadata.get("stop_reason")
    if reason == "end_turn":
        return "done", reason
    if reason == "refusal":
        return "refused", (metadata.get("stop_details") or {}).get("category")
    if reason == "max_tokens":
        return "truncated", reason
    if reason == "model_context_window_exceeded":
        return "overflow", reason
    return "other", reason


def _openai_stop(raw: AIMessage) -> tuple[Stop, str | None]:
    """Итог ответа OpenAI: finish_reason в Chat Completions, status в Responses API"""
    metadata = raw.response_metadata
    incomplete = (metadata.get("incomplete_details") or {}).get("reason")
    reason = metadata.get("finish_reason") or incomplete or metadata.get("status")
    refusal = raw.additional_kwargs.get("refusal") or next(
        (
            block.get("refusal")
            for block in raw.content
            if isinstance(block, dict) and block.get("type") == "refusal"
        ),
        None,
    )
    if refusal:
        return "refused", refusal
    if reason == "content_filter":
        return "refused", reason
    if reason in ("stop", "completed"):
        return "done", reason
    if reason in ("length", "max_output_tokens"):
        return "truncated", reason
    return "other", reason


def _gemini_stop(metadata: dict[str, Any]) -> tuple[Stop, str | None]:
    """Итог ответа Gemini по блокировке запроса и finish_reason"""
    blocked = (metadata.get("prompt_feedback") or {}).get("block_reason")
    blocked = getattr(blocked, "value", blocked)
    if blocked and blocked != "BLOCKED_REASON_UNSPECIFIED":
        return "refused", blocked
    reason = metadata.get("finish_reason")
    if reason == "STOP":
        return "done", reason
    if reason in GEMINI_REFUSALS:
        return "refused", reason
    if reason == "MAX_TOKENS":
        return "truncated", reason
    return "other", reason


def _ollama_stop(metadata: dict[str, Any]) -> tuple[Stop, str | None]:
    """Итог ответа локальной модели Ollama по done_reason; отказов у нее нет"""
    reason = metadata.get("done_reason")
    if reason == "stop":
        return "done", reason
    if reason == "length":
        return "truncated", reason
    return "other", reason


def _parse(raw: AIMessage, schema: type[SchemaT]) -> SchemaT:
    """
    Разбор текста ответа по схеме

    Описание:
        Обычно ответ - один JSON-объект. После отказа Claude посреди ответа
        резервная модель продолжает частичный текст (ответ - весь текст) или
        начинает заново (ответ - окончание текста с одной из открывающих скобок),
        и LangChain может склеить оба текста в одну строку; локальная модель
        с рассуждением по умолчанию пишет его перед ответом в тегах think.
        Поэтому разбирается весь текст, а затем его окончания с каждой
        открывающей скобки
    """
    if isinstance(raw.content, str):
        text = raw.content
    else:
        text = "".join(_block_text(block) for block in raw.content)
    starts = [0] + [i for i, char in enumerate(text) if char == "{" and i > 0]
    for start in starts:
        try:
            return schema.model_validate_json(text[start:])
        except ValidationError as error:
            last_error = error
    raise ModelError(_MISMATCH.format(schema=schema.__name__), retryable=True) from last_error


def _block_text(block: str | dict[Any, Any]) -> str:
    """Текст блока содержимого сообщения LangChain; у блоков не с текстом - пустая строка"""
    if isinstance(block, str):
        return block
    return str(block.get("text", "")) if block.get("type") == "text" else ""


def _incomplete_reason(error: Exception) -> str | None:
    """Причина незавершенного ответа Responses API из HTTP-ответа, прикрепленного LangChain"""
    response = getattr(error, "response", None)
    try:
        body = response.json() if response is not None else {}
    except ValueError:
        return None
    reason = (body.get("incomplete_details") or {}).get("reason")
    return reason if isinstance(reason, str) else None


def _transient(error: Exception) -> bool:
    """
    Временный сбой вызова: перегрузка или сбой сервера, лимит запросов, сеть, таймаут

    Описание:
        Ошибки провайдеров LangChain сам отмечает is_retryable; SDK их уже
        повторил, но повтор узла позже может помочь. Ошибка посреди потока
        Claude приходит со статусом 200 и без такой отметки
    """
    if getattr(error, "is_retryable", False) or isinstance(error, (ConnectionError, TimeoutError)):
        return True
    body = getattr(error, "body", None)
    return (
        getattr(error, "status_code", None) == 200
        and isinstance(body, dict)
        and (body.get("error") or {}).get("type") in TRANSIENT_STREAM_ERRORS
    )


def _model_name(model: BaseChatModel) -> str:
    """Имя модели из ее настроек: у моделей LangChain поле называется по-разному"""
    return str(getattr(model, "model_name", None) or getattr(model, "model", None) or "")
