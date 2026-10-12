import operator
from dataclasses import dataclass
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy

from . import tasks
from .attempts import Attempt, Outcome, Result, best, critique
from .measure import Direction, Measurement, Target, measure
from .model import ModelError
from .tasks import FactCheck


@dataclass(frozen=True)
class Context:
    """
    Модели, которые узлы графа получают через Runtime

    Атрибуты:
        writer (BaseChatModel): Модель писателя
        judge (BaseChatModel): Модель судьи: факты исходника и их проверка
    """

    writer: BaseChatModel
    judge: BaseChatModel


class State(TypedDict):
    """
    Состояние графа

    Атрибуты:
        source (str): Исходный текст
        target (Target): Цель адаптации
        max_attempts (int): Наибольшее число попыток
        source_measurement (Measurement): Уровень и признаки исходника
        facts (list[str]): Факты исходника, выписанные один раз
        facts_tokens (int): Токены судьи на факты
        draft (str): Черновик текущей попытки
        draft_tokens (int): Токены писателя на черновик
        draft_measurement (Measurement): Уровень и признаки черновика
        checks (list[FactCheck]): Проверки фактов в черновике
        judge_tokens (int): Токены судьи на проверку
        attempts (list[Attempt]): Все попытки; узел добавляет к списку, а не заменяет его
        critique (str): Критика последней попытки
        result (Result): Итог
    """

    source: str
    target: Target
    max_attempts: int
    source_measurement: NotRequired[Measurement]
    facts: NotRequired[list[str]]
    facts_tokens: NotRequired[int]
    draft: NotRequired[str]
    draft_tokens: NotRequired[int]
    draft_measurement: NotRequired[Measurement]
    checks: NotRequired[list[FactCheck]]
    judge_tokens: NotRequired[int]
    attempts: NotRequired[Annotated[list[Attempt], operator.add]]
    critique: NotRequired[str]
    result: NotRequired[Result]


def _retryable(error: Exception) -> bool:
    """Повторять узел только при ответе модели, который может получиться при повторе"""
    return isinstance(error, ModelError) and error.retryable


RETRY = RetryPolicy(max_attempts=3, retry_on=_retryable)


def measure_source(state: State) -> dict[str, Any]:
    """Уровень и признаки исходного текста"""
    return {"source_measurement": measure(state["source"])}


def after_source(state: State) -> Literal["finish", "extract_facts"]:
    """Исходник уже на уровне цели - адаптировать нечего"""
    direction = state["target"].direction(state["source_measurement"].grade)
    return "finish" if direction is Direction.KEEP else "extract_facts"


def extract_facts(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Факты исходника: один раз, все черновики сверяются с одним списком"""
    reply = tasks.extract_facts(runtime.context.judge, state["source"])
    return {"facts": reply.output, "facts_tokens": reply.input_tokens + reply.output_tokens}


def rewrite(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Черновик новой попытки: исходник, переписанный с учетом критики прошлой"""
    attempts = state.get("attempts", [])
    reply = tasks.rewrite(
        runtime.context.writer,
        state["source"],
        state["source_measurement"],
        state["target"],
        state["facts"],
        previous=attempts[-1].text if attempts else None,
        critique=state.get("critique"),
    )
    return {"draft": reply.output, "draft_tokens": reply.input_tokens + reply.output_tokens}


def measure_draft(state: State) -> dict[str, Any]:
    """Уровень и признаки черновика; идет параллельно с проверкой фактов"""
    return {"draft_measurement": measure(state["draft"])}


def judge(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Проверка фактов исходника в черновике; идет параллельно с измерением"""
    reply = tasks.judge(runtime.context.judge, state["facts"], state["draft"])
    return {"checks": reply.output, "judge_tokens": reply.input_tokens + reply.output_tokens}


def record(state: State) -> dict[str, Any]:
    """Попытка из черновика, его уровня и проверки фактов, когда готовы обе ветви"""
    attempt = Attempt(
        number=len(state.get("attempts", [])) + 1,
        text=state["draft"],
        measurement=state["draft_measurement"],
        checks=state["checks"],
        gap=state["target"].gap(state["draft_measurement"].grade),
        tokens=state["draft_tokens"] + state["judge_tokens"],
    )
    return {"attempts": [attempt]}


def route(state: State) -> Literal["finish", "critique"]:
    """Принять попытку, закончить по лимиту или критиковать и переписать снова"""
    attempts = state["attempts"]
    if attempts[-1].accepted or len(attempts) >= state["max_attempts"]:
        return "finish"
    return "critique"


def criticize(state: State) -> dict[str, Any]:
    """Критика последней попытки без модели"""
    text = critique(
        state["attempts"][-1], state["source_measurement"], state["target"], state["facts"]
    )
    return {"critique": text}


def finish(state: State) -> dict[str, Any]:
    """Итог: исходник уже на уровне, принятая попытка или лучшая из всех"""
    attempts = state.get("attempts", [])
    if not attempts:
        return {"result": Result(Outcome.ON_LEVEL, state["source"], None)}
    if attempts[-1].accepted:
        return {"result": Result(Outcome.ACCEPTED, attempts[-1].text, attempts[-1])}
    chosen = best(attempts)
    return {"result": Result(Outcome.BEST, chosen.text, chosen)}


def build_graph(retry_policy: RetryPolicy = RETRY) -> CompiledStateGraph[State, Context]:
    """
    Граф адаптации текста

    Описание:
        Исходник измеряется; если он уже на уровне цели, граф заканчивается.
        Иначе судья один раз выписывает факты, и начинается цикл: писатель
        переписывает исходник, черновик параллельно измеряется и проверяется
        судьей, попытка принимается, если она на уровне цели и все факты
        сохранены; после лимита попыток выбирается лучшая, иначе критика без
        модели уходит писателю на следующую попытку. Узлы с моделью повторяются
        при ответе не по схеме

    Аргументы:
        retry_policy (RetryPolicy): Политика повторов узлов с моделью

    Вывод:
        CompiledStateGraph: Граф, который запускается с Context моделей
    """
    graph = StateGraph(State, context_schema=Context)
    graph.add_node("measure_source", measure_source)
    graph.add_node("extract_facts", extract_facts, retry_policy=retry_policy)
    graph.add_node("rewrite", rewrite, retry_policy=retry_policy)
    graph.add_node("measure", measure_draft)
    graph.add_node("judge", judge, retry_policy=retry_policy)
    graph.add_node("record", record)
    graph.add_node("critique", criticize)
    graph.add_node("finish", finish)
    graph.add_edge(START, "measure_source")
    graph.add_conditional_edges("measure_source", after_source)
    graph.add_edge("extract_facts", "rewrite")
    graph.add_edge("rewrite", "measure")
    graph.add_edge("rewrite", "judge")
    graph.add_edge(["measure", "judge"], "record")
    graph.add_conditional_edges("record", route)
    graph.add_edge("critique", "rewrite")
    graph.add_edge("finish", END)
    return graph.compile()
