"""Живые прогоны графа на настоящей модели: только make test-llm, они стоят денег"""

import pytest

from ruts_adapt.attempts import Outcome
from tests.live import CASES, run_graph

pytestmark = pytest.mark.llm


@pytest.mark.parametrize("case", CASES, ids=[case.slug for case in CASES])
def test_live(case, context, live_runs):
    """Граф доходит до итога, и итог ближе к цели, чем исходник"""
    state = run_graph(case, context, live_runs)
    result = state["result"]
    source_gap = state["target"].gap(state["source_measurement"].grade)
    if result.outcome is Outcome.ON_LEVEL:
        assert source_gap == 0
        return
    assert state["facts"]
    assert result.attempt is not None
    assert abs(result.attempt.gap) < abs(source_gap)
    if result.outcome is Outcome.ACCEPTED:
        assert result.attempt.accepted
