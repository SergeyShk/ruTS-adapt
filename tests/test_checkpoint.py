import sqlite3

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from ruts_adapt import checkpoint
from ruts_adapt.attempts import Attempt, Outcome, Result
from ruts_adapt.checkpoint import (
    DATABASE,
    STATE_TYPES,
    ThreadBusyError,
    checkpointer,
    prune,
    serializer,
    thread_lock,
)
from ruts_adapt.graph import Context, build_graph
from ruts_adapt.measure import Measurement, Target
from ruts_adapt.tasks import Draft, Facts, FactStatus
from tests.fakes import FakeChatModel, anthropic
from tests.samples import DECREE, FACTS, MEDIUM, ONE_LOST, SIMPLE, verdict
from tests.test_graph import FAST

CONFIG = {"configurable": {"thread_id": "t"}}


def run_saved(saver):
    """Полный прогон графа с двумя попытками, состояние которого сохранено"""
    writer = FakeChatModel(replies=[anthropic(Draft(text=MEDIUM)), anthropic(Draft(text=SIMPLE))])
    judge = FakeChatModel(
        replies=[
            anthropic(Facts(facts=FACTS)),
            anthropic(ONE_LOST),
            anthropic(verdict(FactStatus.KEPT, FactStatus.DISTORTED)),
        ]
    )
    graph = build_graph(FAST, checkpointer=saver)
    state = {"source": DECREE, "target": Target.from_grade(2), "max_attempts": 2}
    return graph.invoke(state, CONFIG, context=Context(writer=writer, judge=judge))


def test_roundtrip(data_dir):
    """Состояние из базы - те же объекты тех же классов, что и в памяти"""
    with checkpointer(data_dir) as saver:
        state = run_saved(saver)
    assert (data_dir / DATABASE).is_file()
    with checkpointer(data_dir) as saver:
        restored = build_graph(checkpointer=saver).get_state(CONFIG).values
    assert restored == state
    assert isinstance(restored["target"], Target)
    assert isinstance(restored["source_measurement"], Measurement)
    assert isinstance(restored["attempts"][0], Attempt)
    assert isinstance(restored["result"], Result)
    assert restored["result"].outcome is Outcome.BEST
    assert restored["attempts"][1].checks[1].status is FactStatus.DISTORTED


def test_allowlist(data_dir):
    """Без списка STATE_TYPES строгий сериализатор возвращает не те классы"""
    with checkpointer(data_dir) as saver:
        state = run_saved(saver)
        strict = JsonPlusSerializer(allowed_msgpack_modules=None)
        kind, data = saver.serde.dumps_typed(state)
        assert strict.loads_typed((kind, data)) != state
        assert serializer().loads_typed((kind, data)) == state
    assert {kind.__module__.split(".")[0] for kind in STATE_TYPES} == {"ruts_adapt"}


def test_prune(data_dir):
    """Чистка оставляет последний чекпоинт: состояние то же; у неизвестного потока - ничего"""
    with checkpointer(data_dir) as saver:
        state = run_saved(saver)
        prune(saver, "t")
        prune(saver, "unknown")
        (count,) = saver.conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()
        assert count == 1
        assert build_graph(checkpointer=saver).get_state(CONFIG).values == state


def test_thread_lock(data_dir):
    """Поток открывает один процесс за раз; после выхода из контекста блокировка снята"""
    with thread_lock(data_dir, "t"):
        busy = pytest.raises(ThreadBusyError, match=r"^поток t уже выполняется в другом процессе$")
        with busy, thread_lock(data_dir, "t"):
            pass
        with thread_lock(data_dir, "other"):
            pass
    with thread_lock(data_dir, "t"):
        pass


@pytest.mark.parametrize(
    ("errors", "timeout", "raised"),
    [
        (["database is locked"], 5.0, None),
        (["database is locked"], 0.0, "database is locked"),
        (["disk I/O error"], 5.0, "disk I/O error"),
    ],
    ids=["retried", "timeout", "other-error"],
)
def test_setup_retry(data_dir, monkeypatch, errors, timeout, raised):
    """Таблицы новой базы создает другой процесс: повтор до SETUP_TIMEOUT, другие ошибки - сразу"""
    real_setup = SqliteSaver.setup

    def setup(self):
        if errors:
            raise sqlite3.OperationalError(errors.pop(0))
        real_setup(self)

    monkeypatch.setattr(SqliteSaver, "setup", setup)
    monkeypatch.setattr(checkpoint, "SETUP_TIMEOUT", timeout)
    if raised is None:
        with checkpointer(data_dir) as saver:
            assert saver.is_setup
        return
    with pytest.raises(sqlite3.OperationalError, match=raised), checkpointer(data_dir):
        pass
