import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, RetryPolicy

from ruts_adapt.attempts import Outcome
from ruts_adapt.graph import Context, _retryable, build_graph
from ruts_adapt.measure import Target, measure
from ruts_adapt.model import ModelError
from ruts_adapt.tasks import Draft, Facts, FactStatus, Verdict
from tests.fakes import FakeChatModel, anthropic
from tests.samples import DECREE, FACTS, KEPT, MEDIUM, ONE_LOST, SIMPLE, verdict

FAST = RetryPolicy(initial_interval=0.0, jitter=False, retry_on=_retryable)


def models(drafts=(), verdicts=(), facts=FACTS):
    writer = FakeChatModel(replies=[anthropic(Draft(text=draft)) for draft in drafts])
    judge = FakeChatModel(
        replies=[anthropic(Facts(facts=facts))] + [anthropic(item) for item in verdicts]
    )
    return writer, judge


def run(writer, judge, grade=2, max_attempts=3, source=DECREE):
    state = {"source": source, "target": Target.from_grade(grade), "max_attempts": max_attempts}
    return build_graph(FAST).invoke(state, context=Context(writer=writer, judge=judge))


def test_on_level():
    """Исходник уже на уровне цели: ни одного вызова модели"""
    writer, judge = models()
    state = run(writer, judge, grade=2, source=SIMPLE)
    assert state["result"].outcome is Outcome.ON_LEVEL
    assert state["result"].text == SIMPLE
    assert writer.requests == judge.requests == []
    assert "facts" not in state


def test_accepted_first():
    writer, judge = models(drafts=[SIMPLE], verdicts=[KEPT])
    state = run(writer, judge)
    result = state["result"]
    assert (result.outcome, result.text, result.attempt.number) == (Outcome.ACCEPTED, SIMPLE, 1)
    assert state["facts"] == FACTS
    assert state["source_measurement"] == measure(DECREE)
    (attempt,) = state["attempts"]
    assert attempt.measurement == measure(SIMPLE)
    assert attempt.checks == KEPT.checks
    assert attempt.tokens == 165 + 165
    assert "<критика>" not in writer.requests[0][1].content


def test_accepted_after_critique():
    """Вторая попытка видит прошлый черновик и критику без модели и принимается"""
    writer, judge = models(drafts=[MEDIUM, SIMPLE], verdicts=[ONE_LOST, KEPT])
    state = run(writer, judge)
    assert state["result"].outcome is Outcome.ACCEPTED
    assert [attempt.number for attempt in state["attempts"]] == [1, 2]
    assert state["result"].attempt is state["attempts"][1]
    prompt = writer.requests[1][1].content
    assert f"<прошлая_попытка>\n{MEDIUM}\n</прошлая_попытка>" in prompt
    assert f"<критика>\n{state['critique']}\n</критика>" in prompt
    assert "Потеряны факты, верните их:\n2. Субсидии возмещают недополученные доходы" in prompt
    assert len(judge.requests) == 3


def test_limit_best():
    """После лимита попыток выбирается лучшая: без потерь важнее уровня"""
    lossy = verdict(FactStatus.KEPT, FactStatus.LOST)
    writer, judge = models(drafts=[MEDIUM, SIMPLE], verdicts=[KEPT, lossy])
    state = run(writer, judge, max_attempts=2)
    result = state["result"]
    assert result.outcome is Outcome.BEST
    assert (result.attempt.number, result.text) == (1, MEDIUM)
    assert len(state["attempts"]) == 2


def test_retry():
    """Неполный вердикт судьи повторяется, параллельное измерение не повторяется"""
    incomplete = Verdict(checks=KEPT.checks[:1])
    writer, judge = models(drafts=[SIMPLE], verdicts=[incomplete, KEPT])
    state = run(writer, judge)
    assert state["result"].outcome is Outcome.ACCEPTED
    assert len(judge.requests) == 3
    assert len(writer.requests) == 1


def test_retry_exhausted():
    incomplete = Verdict(checks=KEPT.checks[:1])
    writer, judge = models(drafts=[SIMPLE], verdicts=[incomplete] * 3)
    with pytest.raises(ModelError, match="ровно один раз"):
        run(writer, judge)
    assert len(judge.requests) == 4


def test_no_retry_on_refusal():
    """Отказ повторится и при повторе, поэтому узел не повторяется"""
    writer = FakeChatModel(replies=[anthropic(stop_reason="refusal", category="bio")])
    judge = FakeChatModel(replies=[anthropic(Facts(facts=FACTS))])
    with pytest.raises(ModelError, match="отказалась отвечать"):
        run(writer, judge)
    assert len(writer.requests) == 1


def test_stream_updates():
    """Измерение и проверка фактов идут в одном шаге, запись попытки - после обоих"""
    writer, judge = models(drafts=[SIMPLE], verdicts=[KEPT])
    state = {"source": DECREE, "target": Target.from_grade(2), "max_attempts": 3}
    steps = [
        set(update)
        for update in build_graph(FAST).stream(
            state, context=Context(writer=writer, judge=judge), stream_mode="updates"
        )
    ]
    assert steps[:3] == [{"measure_source"}, {"extract_facts"}, {"rewrite"}]
    assert sorted(map(sorted, steps[3:5])) == [["judge"], ["measure"]]
    assert steps[5:] == [{"record"}, {"finish"}]


def test_retryable():
    assert _retryable(ModelError("схема", retryable=True))
    assert not _retryable(ModelError("отказ"))
    assert not _retryable(ValueError("другое"))


def saved(writer, judge, grade=2, max_attempts=3, review=True):
    """Граф с чекпоинтером в памяти и первый запуск потока"""
    graph = build_graph(FAST, checkpointer=InMemorySaver())
    state = {
        "source": DECREE,
        "target": Target.from_grade(grade),
        "max_attempts": max_attempts,
        "review": review,
    }
    context = Context(writer=writer, judge=judge)
    graph.invoke(state, CONFIG, context=context)
    return graph, context


CONFIG = {"configurable": {"thread_id": "t"}}


def test_review_approve():
    """С review принятая попытка ждет решения человека, а не становится итогом"""
    writer, judge = models(drafts=[SIMPLE], verdicts=[KEPT])
    graph, context = saved(writer, judge)
    snapshot = graph.get_state(CONFIG)
    assert snapshot.next == ("review",)
    (pause,) = snapshot.interrupts
    assert pause.value == {"attempt": 1, "outcome": Outcome.ACCEPTED}
    assert "result" not in snapshot.values
    state = graph.invoke(Command(resume={"feedback": None}), CONFIG, context=context)
    assert (state["result"].outcome, state["result"].text) == (Outcome.ACCEPTED, SIMPLE)
    assert graph.get_state(CONFIG).next == ()


def test_review_reject():
    """Возвращенный кандидат переписывается с замечанием, даже сверх лимита попыток"""
    writer, judge = models(drafts=[SIMPLE, SIMPLE + " Это все."], verdicts=[KEPT, KEPT])
    graph, context = saved(writer, judge, max_attempts=1)
    graph.invoke(Command(resume={"feedback": "короче"}), CONFIG, context=context)
    snapshot = graph.get_state(CONFIG)
    assert snapshot.next == ("review",)
    assert snapshot.interrupts[0].value["attempt"] == 2
    assert snapshot.values["feedback"] is None
    prompt = writer.requests[1][1].content
    assert f"<прошлая_попытка>\n{SIMPLE}\n</прошлая_попытка>" in prompt
    assert "Замечание проверяющего: короче\n</критика>" in prompt


def test_review_best_candidate():
    """После лимита на решение уходит лучшая попытка, и замечание относится к ней"""
    lossy = verdict(FactStatus.KEPT, FactStatus.LOST)
    writer, judge = models(drafts=[MEDIUM, SIMPLE, SIMPLE], verdicts=[KEPT, lossy, KEPT])
    graph, context = saved(writer, judge, max_attempts=2)
    assert graph.get_state(CONFIG).interrupts[0].value == {"attempt": 1, "outcome": Outcome.BEST}
    graph.invoke(Command(resume={"feedback": "проще"}), CONFIG, context=context)
    assert f"<прошлая_попытка>\n{MEDIUM}\n</прошлая_попытка>" in writer.requests[2][1].content


def test_resume_after_error():
    """После ошибки модели поток продолжается с упавшего узла: факты не выписываются заново"""
    writer = FakeChatModel(replies=[anthropic(Draft(text=SIMPLE))])
    judge = FakeChatModel(
        replies=[anthropic(Facts(facts=FACTS)), anthropic(stop_reason="refusal")]
    )
    graph = build_graph(FAST, checkpointer=InMemorySaver())
    state = {"source": DECREE, "target": Target.from_grade(2), "max_attempts": 3}
    context = Context(writer=writer, judge=judge)
    with pytest.raises(ModelError):
        graph.invoke(state, CONFIG, context=context)
    judge.replies.append(anthropic(KEPT))
    result = graph.invoke(None, CONFIG, context=context)["result"]
    assert result.outcome is Outcome.ACCEPTED
    assert (len(writer.requests), len(judge.requests)) == (1, 3)
