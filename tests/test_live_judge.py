"""Стенд калибровки судьи на настоящей модели: только make test-llm, он стоит денег"""

import pytest

from ruts_adapt.tasks import FactStatus
from tests.live import CASES, run_bench

pytestmark = pytest.mark.llm


@pytest.mark.parametrize("case", CASES, ids=[case.slug for case in CASES])
def test_judge(case, context, judge_runs):
    """Судья замечает хотя бы один испорченный факт"""
    run = run_bench(case, context, judge_runs)
    assert run.checks is not None
    assert any(
        check.status is not FactStatus.KEPT for check in run.checks if check.fact in run.plan
    )
