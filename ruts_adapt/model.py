from dataclasses import dataclass
from typing import Generic, TypeVar

from anthropic import Anthropic, omit, transform_schema
from anthropic.types.beta import BetaMessage
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
        input_tokens (int): Токены запроса по всем попыткам, включая отказавшую модель
        output_tokens (int): Токены ответа вместе с мышлением по всем попыткам
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
        модель отказалась посреди ответа и ответила резервная, разбирается ее
        ответ, а если он продолжает частичный текст основной - оба вместе

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
    with client.beta.messages.stream(
        model=settings.model,
        max_tokens=settings.max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_config={
            "effort": settings.effort,
            "format": {"type": "json_schema", "schema": transform_schema(schema)},
        },
        fallbacks="default" if settings.fallbacks else omit,
        betas=[FALLBACKS_BETA] if settings.fallbacks else omit,
    ) as stream:
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
    return Reply(_parse(message, schema), message.model, *_tokens(message))


def _parse(message: BetaMessage, schema: type[SchemaT]) -> SchemaT:
    """Разбор текста ответа по схеме"""
    switches = [i for i, block in enumerate(message.content) if block.type == "fallback"]
    texts = ["".join(block.text for block in message.content if block.type == "text")]
    if switches:
        after = message.content[switches[-1] + 1 :]
        # Резервная модель может начать ответ заново или продолжить частичный текст отказавшей
        texts.insert(0, "".join(block.text for block in after if block.type == "text"))
    for text in texts:
        try:
            return schema.model_validate_json(text)
        except ValidationError as error:
            last_error = error
    raise ModelError(f"Ответ модели не соответствует схеме {schema.__name__}") from last_error


def _tokens(message: BetaMessage) -> tuple[int, int]:
    """Токены запроса и ответа по всем попыткам: верхний usage считает только ответившую"""
    attempts = [
        usage
        for usage in message.usage.iterations or []
        if usage.type in ("message", "fallback_message")
    ]
    if not attempts:
        return message.usage.input_tokens, message.usage.output_tokens
    input_tokens = sum(usage.input_tokens for usage in attempts)
    output_tokens = sum(usage.output_tokens for usage in attempts)
    return input_tokens, output_tokens
