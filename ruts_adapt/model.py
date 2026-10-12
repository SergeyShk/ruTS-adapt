from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from anthropic import Anthropic, transform_schema
from pydantic import BaseModel, ValidationError

from .settings import ModelSettings

FALLBACKS_BETA = "server-side-fallback-2026-07-01"

T = TypeVar("T")
U = TypeVar("U")
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class ModelError(Exception):
    """Ответ модели, который нельзя использовать: отказ, обрезанный или не по схеме"""


@dataclass(frozen=True)
class Reply(Generic[T]):
    """
    Ответ модели

    Атрибуты:
        output (Any): Разобранный ответ
        model (str): Модель, которая ответила; при отказе основной - резервная
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


def ask(
    client: Anthropic, settings: ModelSettings, system: str, prompt: str, schema: type[SchemaT]
) -> Reply[SchemaT]:
    """
    Запрос к модели с ответом по схеме

    Описание:
        Ответ ограничен JSON-схемой Pydantic-модели (structured outputs) и приходит
        потоком. Причина остановки проверяется до разбора: отказ и обрезанный
        на max_tokens ответ - ошибка, а не пустой результат. Если основная
        модель отказалась и ответила резервная, разбирается ответ резервной

    Аргументы:
        client (Anthropic): Клиент API
        settings (ModelSettings): Модель, усилие, max_tokens и резервная модель
        system (str): Системный промпт
        prompt (str): Сообщение пользователя
        schema (type[BaseModel]): Схема ответа

    Вывод:
        Reply: Ответ, разобранный по схеме

    Исключения:
        ModelError: Если модель отказалась отвечать, ответ обрезан на max_tokens,
            остановился по другой причине или не соответствует схеме
    """
    request: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {
            "effort": settings.effort,
            "format": {"type": "json_schema", "schema": transform_schema(schema)},
        },
    }
    if settings.fallbacks:
        request |= {"fallbacks": "default", "betas": [FALLBACKS_BETA]}
    with client.beta.messages.stream(**request) as stream:
        message = stream.get_final_message()
    if message.stop_reason == "refusal":
        category = message.stop_details.category if message.stop_details else None
        raise ModelError(
            f"Модель {message.model} отказалась отвечать (категория отказа: {category or 'не указана'})"
        )
    if message.stop_reason == "max_tokens":
        raise ModelError(
            f"Ответ модели обрезан на max_tokens={settings.max_tokens}: увеличьте лимит в настройках"
        )
    if message.stop_reason != "end_turn":
        raise ModelError(f"Модель остановилась по неожиданной причине: {message.stop_reason}")
    # Отказ посреди потока оставляет частичный текст первой модели до блока fallback
    switches = [i for i, block in enumerate(message.content) if block.type == "fallback"]
    blocks = message.content[switches[-1] + 1 :] if switches else message.content
    text = "".join(block.text for block in blocks if block.type == "text")
    try:
        output = schema.model_validate_json(text)
    except ValidationError as error:
        raise ModelError(f"Ответ модели не соответствует схеме {schema.__name__}") from error
    return Reply(output, message.model, message.usage.input_tokens, message.usage.output_tokens)
