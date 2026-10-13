"""Прогоны и отчет живых тестов на поддельных моделях: платный запуск не должен упасть на них"""

import pytest

from ruts_adapt.attempts import Outcome
from ruts_adapt.graph import Context
from ruts_adapt.model import ModelError
from ruts_adapt.tasks import Draft, FactCheck, Facts, FactStatus, Verdict
from tests.fakes import USAGE, FakeChatModel, RepliesExhaustedError, anthropic
from tests.live import Case, Damaged, damage_plan, retried, run_bench, run_graph
from tests.live_report import JudgeRun, details, summary
from tests.samples import DECREE, FACTS, KEPT, MEDIUM, ONE_LOST, SIMPLE

CASE = Case("decree", "Постановление", "официальный документ", lambda: DECREE, 2)
CALL = USAGE["input_tokens"] + USAGE["output_tokens"]
BENCH_FACTS = [f"Факт номер {number}" for number in range(1, 13)]
DAMAGED = Damaged(text="Пересказ с порчей.\nВторой абзац.", changes=["2: убран", "4: число"])
# Порча damage_plan(12): убраны 2, 6, 10, искажены 4, 8. Судья пропустил искажение 4,
# назвал искаженным убранный 6 и поднял ложную тревогу на 3
BENCH_VERDICT = {
    2: FactStatus.LOST,
    3: FactStatus.LOST,
    6: FactStatus.DISTORTED,
    8: FactStatus.DISTORTED,
    10: FactStatus.LOST,
}


def bench_models(*writer_replies):
    writer = FakeChatModel(replies=list(writer_replies))
    checks = []
    for number in range(1, 13):
        status = BENCH_VERDICT.get(number, FactStatus.KEPT)
        quote = "" if status is FactStatus.LOST else f"цитата {number}"
        checks.append(FactCheck(fact=number, status=status, quote=quote))
    judge = FakeChatModel(
        replies=[anthropic(Facts(facts=BENCH_FACTS)), anthropic(Verdict(checks=checks))]
    )
    return writer, judge


def graph_run():
    """Прогон графа в две попытки: в первой потерян факт"""
    writer = FakeChatModel(replies=[anthropic(Draft(text=MEDIUM)), anthropic(Draft(text=SIMPLE))])
    judge = FakeChatModel(
        replies=[anthropic(Facts(facts=FACTS)), anthropic(ONE_LOST), anthropic(KEPT)]
    )
    runs = []
    state = run_graph(CASE, Context(writer=writer, judge=judge), runs)
    return state, runs


def test_run_graph():
    """Прогон записывает попытки, критику к ним, итог и токены всех вызовов"""
    state, runs = graph_run()
    (run,) = runs
    assert state["result"].outcome is Outcome.ACCEPTED
    assert run.case == "Постановление"
    assert run.facts == FACTS
    assert [attempt.text for attempt in run.attempts] == [MEDIUM, SIMPLE]
    assert len(run.critiques) == 1
    assert run.critiques[0].startswith("Попытка 1 читается как")
    assert run.result == state["result"]
    assert run.tokens == 5 * CALL
    assert run.error is None


def test_run_graph_error():
    """Граф упал: в отчет попадает все, что сделано до ошибки"""
    judge = FakeChatModel(replies=[anthropic(Facts(facts=FACTS))])
    runs = []
    with pytest.raises(RepliesExhaustedError):
        run_graph(CASE, Context(writer=FakeChatModel(replies=[]), judge=judge), runs)
    (run,) = runs
    assert run.facts == FACTS
    assert run.attempts == run.critiques == []
    assert run.result is None
    assert run.tokens == CALL
    assert run.error == "RepliesExhaustedError: Поддельной модели не задан ответ на запрос"


def test_run_bench():
    """Писатель получает номера для порчи, вердикт судьи сверяется с ними"""
    writer, judge = bench_models(anthropic(DAMAGED))
    runs = []
    run = run_bench(CASE, Context(writer=writer, judge=judge), runs)
    assert runs == [run]
    assert run.plan == damage_plan(12)
    assert (run.text, run.changes) == (DAMAGED.text, DAMAGED.changes)
    prompt = writer.requests[0][1].content
    assert prompt.startswith("Перескажи исходный текст для читателя уровня «1-3-й класс")
    assert f"<исходный_текст>\n{DECREE}\n</исходный_текст>" in prompt
    assert "1. Факт номер 1\n2. Факт номер 2\n" in prompt
    assert prompt.endswith("<задание>\nУбрать: 2, 6, 10\nИсказить: 4, 8\n</задание>")
    assert judge.requests[1][1].content.endswith(f"<пересказ>\n{DAMAGED.text}\n</пересказ>")
    assert run.matrix() == {
        (FactStatus.LOST, FactStatus.LOST): 2,
        (FactStatus.LOST, FactStatus.DISTORTED): 1,
        (FactStatus.DISTORTED, FactStatus.KEPT): 1,
        (FactStatus.DISTORTED, FactStatus.DISTORTED): 1,
        (FactStatus.KEPT, FactStatus.LOST): 1,
        (FactStatus.KEPT, FactStatus.KEPT): 6,
    }
    assert run.tokens == 3 * CALL


def test_run_bench_retry():
    """Ответ не по схеме повторяется, как в узле графа"""
    writer, judge = bench_models(anthropic("не JSON"), anthropic(DAMAGED))
    run = run_bench(CASE, Context(writer=writer, judge=judge), [])
    assert run.text == DAMAGED.text
    assert len(writer.requests) == 2


def test_run_bench_error():
    """Отказ писателя не повторяется: в отчете факты и ошибка, пересказа и вердикта нет"""
    writer, judge = bench_models(anthropic(stop_reason="refusal", category="cyber"))
    runs = []
    with pytest.raises(ModelError, match="отказалась"):
        run_bench(CASE, Context(writer=writer, judge=judge), runs)
    (run,) = runs
    assert run.facts == BENCH_FACTS
    assert (run.text, run.changes, run.checks) == (None, [], None)
    assert run.error.startswith("ModelError: ")
    assert run.matrix() == {}


def test_retried():
    """Повторяемая ошибка - до трех вызовов, затем она пробрасывается"""
    calls = []

    def failing():
        calls.append(1)
        raise ModelError("не по схеме", retryable=True)

    with pytest.raises(ModelError):
        retried(failing)
    assert len(calls) == 3


def test_summary():
    _, runs = graph_run()
    writer, judge = bench_models(anthropic(DAMAGED))
    bench = run_bench(CASE, Context(writer=writer, judge=judge), [])
    failed = JudgeRun("Сломанный", BENCH_FACTS, {}, None, [], None, CALL, "ModelError: отказ")
    lines = summary(runs, [bench, failed]).splitlines()
    assert lines[0] == "## Прогоны графа"
    assert lines[4].startswith(
        "| Постановление | официальный документ | 1-3-й класс (6-8 лет) | 26.5 | 25.5 → 1.0 "
        "| 1/0/1 → 2/0/0 (из 2) | принят, попытка 2 |"
    )
    assert lines[4].endswith(f"| {5 * CALL} |")
    assert lines[6:] == [
        "## Стенд судьи",
        "",
        "| Текст | Фактов | Убраны: с/и/п | Искажены: с/и/п | Не тронуты: с/и/п | Токены |",
        "| :-- | :-: | :-: | :-: | :-: | :-: |",
        f"| Постановление | 12 | 0/1/2 | 1/1/0 | 6/0/1 | {3 * CALL} |",
        f"| Сломанный | 12 | ошибка: ModelError: отказ | |  | {CALL} |",
        f"| Всего | | 0/1/2 | 1/1/0 | 6/0/1 | {4 * CALL} |",
        "",
        "Судья заметил порчу: 4 из 5 (80%), с верным статусом: 3 из 5 (60%); "
        "ложные тревоги: 1 из 7 (14%) нетронутых фактов",
    ]
    assert summary([], []) == ""


def test_details():
    """Тексты: абзацы через пустую строку, вердикты и критика к попыткам, расхождения судьи"""
    _, runs = graph_run()
    writer, judge = bench_models(anthropic(DAMAGED))
    bench = run_bench(CASE, Context(writer=writer, judge=judge), [])
    failed = JudgeRun("Сломанный", [], {}, None, [], None, 0, "ModelError: отказ")
    text = details(runs, [bench, failed])
    sections = text.split("\n\n")
    assert sections[:5] == [
        "## Тексты прогонов графа",
        "### Постановление: 1-3-й класс (6-8 лет)",
        "Исходник: класс 26.5, 18.0 слова в предложении",
        DECREE,
        "1. Постановление устанавливает порядок субсидий\n"
        "2. Субсидии возмещают недополученные доходы",
    ]
    assert sections[5].startswith("#### Попытка 1: класс 25.5, ")
    assert sections[5].endswith("факты 1/0/1")
    assert sections[6:8] == [MEDIUM, "- 2. Субсидии возмещают недополученные доходы - потерян"]
    assert sections[8:10] == ["Критика:", f"```text\n{runs[0].critiques[0]}\n```"]
    assert "#### Попытка 2: класс 1.0" in text
    judge_part = text[text.index("## Тексты стенда судьи") :].split("\n\n")
    assert judge_part == [
        "## Тексты стенда судьи",
        "### Постановление",
        "Пересказ с порчей.",
        "Второй абзац.",
        "Порча со слов писателя:",
        "- 2: убран\n- 4: число",
        "Расхождения судьи с порчей:",
        "- 3. Факт номер 3 - писатель: сохранен, судья: потерян\n"
        "- 4. Факт номер 4 - писатель: искажен, судья: сохранен: «цитата 4»\n"
        "- 6. Факт номер 6 - писатель: потерян, судья: искажен: «цитата 6»",
        "### Сломанный",
        "Ошибка: ModelError: отказ",
    ]
    assert details([], []) == ""
