import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

from anthropic import omit
from anthropic.resources.beta.messages import Messages
from anthropic.types.beta import BetaMessage
from pydantic import BaseModel

STREAM = inspect.signature(Messages.stream)


def message(
    output: BaseModel | str,
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5-5",
    category: str | None = None,
    partial: str | None = None,
    iterations: list[dict[str, Any]] | None = None,
) -> BetaMessage:
    """
    Ответ API: пустой блок мышления, как при скрытом мышлении, и текст с JSON

    С partial ответ повторяет отказ основной модели посреди потока: ее частичный
    текст, блок fallback и ответ резервной модели model; iterations - токены
    отдельных попыток, как их отдает API при резервной модели
    """
    text = output if isinstance(output, str) else output.model_dump_json()
    refusal = {"type": "refusal", "category": category, "explanation": None}
    content: list[dict[str, Any]] = [{"type": "thinking", "thinking": "", "signature": "test"}]
    if partial is not None:
        content += [
            {"type": "text", "text": partial},
            {
                "type": "fallback",
                "from": {"model": "claude-opus-5-5"},
                "to": {"model": model},
                "trigger": {"type": "refusal", "category": "cyber"},
            },
        ]
    content.append({"type": "text", "text": text})
    if iterations is not None:
        cache = {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        iterations = [cache | iteration for iteration in iterations]
    return BetaMessage.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_details": refusal if stop_reason == "refusal" else None,
            "stop_sequence": None,
            "usage": {"input_tokens": 120, "output_tokens": 45, "iterations": iterations},
        }
    )


class FakeClient:
    """Поддельный клиент API: отдает заданные ответы по очереди и запоминает запросы"""

    def __init__(self, *replies: BetaMessage) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    @contextmanager
    def _stream(self, **request: Any) -> Iterator[SimpleNamespace]:
        assert self.replies, "Поддельному клиенту не задан ответ на запрос"
        # Запрос, который не примет настоящий SDK, падает здесь с той же TypeError
        STREAM.bind(None, **request)
        self.requests.append({key: value for key, value in request.items() if value is not omit})
        reply = self.replies.pop(0)
        yield SimpleNamespace(get_final_message=lambda: reply)
