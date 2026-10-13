import operator
from dataclasses import dataclass
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, interrupt

from . import tasks
from .attempts import Attempt, Outcome, Result, candidate, critique
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
        review (bool): Остановиться перед итогом и ждать решения человека
        source_measurement (Measurement): Уровень и признаки исходника
        facts (list[str]): Факты исходника, выписанные один раз
        draft (str): Черновик текущей попытки
        draft_tokens (int): Токены писателя на черновик
        draft_measurement (Measurement): Уровень и признаки черновика
        checks (list[FactCheck]): Проверки фактов в черновике
        judge_tokens (int): Токены судьи на проверку
        attempts (list[Attempt]): Все попытки; узел добавляет к списку, а не заменяет его
        critique (str): Критика попытки, по которой пишется следующая
        previous (str): Черновик этой попытки
        feedback (str | None): Замечание человека к кандидату на итог; None - принят
        remarks (list[str]): Все замечания человека; узел добавляет к списку
        rejected (list[int]): Номера попыток, которые вернул человек; узел добавляет к списку
        rejected_after (int): Число попыток при последнем возврате: итог - только из более поздних
        version (int): Версия формата сохраненного состояния
        result (Result): Итог
    """

    source: str
    target: Target
    max_attempts: int
    review: NotRequired[bool]
    source_measurement: NotRequired[Measurement]
    facts: NotRequired[list[str]]
    draft: NotRequired[str]
    draft_tokens: NotRequired[int]
    draft_measurement: NotRequired[Measurement]
    checks: NotRequired[list[FactCheck]]
    judge_tokens: NotRequired[int]
    attempts: NotRequired[Annotated[list[Attempt], operator.add]]
    critique: NotRequired[str]
    previous: NotRequired[str]
    feedback: NotRequired[str | None]
    remarks: NotRequired[Annotated[list[str], operator.add]]
    rejected: NotRequired[Annotated[list[int], operator.add]]
    rejected_after: NotRequired[int]
    version: NotRequired[int]
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
    return {"facts": reply.output}


def rewrite(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Черновик новой попытки: исходник, переписанный с учетом критики прошлой"""
    reply = tasks.rewrite(
        runtime.context.writer,
        state["source"],
        state["source_measurement"],
        state["target"],
        state["facts"],
        previous=state.get("previous"),
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


def route(state: State) -> Literal["finish", "review", "critique"]:
    """Принять попытку, закончить по лимиту или критиковать и переписать снова"""
    attempts = state["attempts"]
    if attempts[-1].accepted or len(attempts) >= state["max_attempts"]:
        return "review" if state.get("review") else "finish"
    return "critique"


def review(state: State) -> dict[str, Any]:
    """
    Пауза перед итогом: человек принимает кандидата или возвращает его с замечанием

    Описание:
        Граф останавливается (interrupt) и сохраняет состояние; продолжение
        передает {"feedback": None}, чтобы принять, или {"feedback": "замечание"},
        чтобы переписать кандидата с этим замечанием, даже сверх лимита попыток
    """
    result = candidate(state["attempts"], state.get("rejected_after", 0))
    assert result.attempt is not None
    decision = interrupt({"attempt": result.attempt.number, "outcome": result.outcome})
    feedback = decision["feedback"]
    if not feedback:
        return {"feedback": None}
    return {
        "feedback": feedback,
        "remarks": [feedback],
        "rejected": [result.attempt.number],
        "rejected_after": len(state["attempts"]),
    }


def after_review(state: State) -> Literal["finish", "critique"]:
    """Принятый кандидат - итог, возвращенный - на новую попытку"""
    return "critique" if state.get("feedback") else "finish"


def criticize(state: State) -> dict[str, Any]:
    """
    Критика без модели: последней попытки или кандидата, возвращенного человеком

    Описание:
        Замечания человека повторяются в каждой следующей критике, чтобы писатель
        учитывал их и в попытках после той, что сразу за возвратом
    """
    attempts = state["attempts"]
    if state.get("feedback"):
        number = state["rejected"][-1]
        attempt = next(attempt for attempt in attempts if attempt.number == number)
    else:
        attempt = attempts[-1]
    text = critique(
        attempt, attempts, state["source_measurement"], state["target"], state["facts"]
    )
    for remark in state.get("remarks", []):
        text += f"\nЗамечание проверяющего: {remark}"
    return {"critique": text, "previous": attempt.text, "feedback": None}


def finish(state: State) -> dict[str, Any]:
    """Итог: исходник уже на уровне, принятая попытка или лучшая после последнего возврата"""
    attempts = state.get("attempts", [])
    if not attempts:
        return {"result": Result(Outcome.ON_LEVEL, state["source"], None)}
    return {"result": candidate(attempts, state.get("rejected_after", 0))}


def build_graph(
    retry_policy: RetryPolicy = RETRY, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> CompiledStateGraph[State, Context]:
    """
    Граф адаптации текста

    Описание:
        Исходник измеряется; если он уже на уровне цели, граф заканчивается.
        Иначе судья один раз выписывает факты, и начинается цикл: писатель
        переписывает исходник, черновик параллельно измеряется и проверяется
        судьей, попытка принимается, если она на уровне цели и все факты
        сохранены; после лимита попыток выбирается лучшая, иначе критика без
        модели уходит писателю на следующую попытку. Узлы с моделью повторяются
        при ответе не по схеме. С review в состоянии граф перед итогом ждет
        решения человека (review), с чекпоинтером - сохраняет состояние после
        каждого шага и продолжается с места остановки

    Аргументы:
        retry_policy (RetryPolicy): Политика повторов узлов с моделью
        checkpointer (BaseCheckpointSaver): Хранилище состояния; без него граф
            не продолжается и не ждет решения человека

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
    graph.add_node("review", review)
    graph.add_node("finish", finish)
    graph.add_edge(START, "measure_source")
    graph.add_conditional_edges("measure_source", after_source)
    graph.add_edge("extract_facts", "rewrite")
    graph.add_edge("rewrite", "measure")
    graph.add_edge("rewrite", "judge")
    graph.add_edge(["measure", "judge"], "record")
    graph.add_conditional_edges("record", route)
    graph.add_conditional_edges("review", after_review)
    graph.add_edge("critique", "rewrite")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)
