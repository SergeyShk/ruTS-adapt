"""Схема графа в README совпадает с графом в коде"""

import re
from collections import Counter
from pathlib import Path

import pytest
from langgraph.graph import END, START

from ruts_adapt.graph import build_graph

ROOT = Path(__file__).parent.parent
ENDS = {"start": START, "done": END}
# Любая связь Mermaid: стрелки, линии и пунктир всех видов
LINK = re.compile(r"--|==|-\.|~~~")
EDGE = re.compile(r"\s*(\w+)\s*(-->|-\.->)\s*(\w+)\s*")


def diagram_edges(text: str) -> Counter[tuple[str, str, bool]]:
    """
    Ребра блока mermaid: откуда, куда, условное ли (пунктир)

    Подписи в кавычках вырезаются; строка со связью в любой другой записи,
    кроме «узел стрелка узел», - ошибка, чтобы ребро не ускользнуло от сверки
    """
    (block,) = re.findall(r"```mermaid\n(.*?)```", text, re.DOTALL)
    edges: Counter[tuple[str, str, bool]] = Counter()
    for line in block.splitlines():
        bare = re.sub(r'\|"[^"]*"\||[\[({]+"[^"]*"[\])}]+', "", line)
        if not LINK.search(bare):
            continue
        match = EDGE.fullmatch(bare)
        assert match, f"ребро в неразобранной записи: {line.strip()}"
        source, arrow, target = match.groups()
        edges[ENDS.get(source, source), ENDS.get(target, target), arrow == "-.->"] += 1
    return edges


@pytest.mark.parametrize("name", ["README.md", "README.en.md"])
def test_diagram_matches_graph(name):
    graph = build_graph().get_graph()
    expected = Counter((edge.source, edge.target, edge.conditional) for edge in graph.edges)
    assert diagram_edges((ROOT / name).read_text(encoding="utf-8")) == expected


@pytest.mark.parametrize(
    "line",
    [
        "    critique --> rewrite --> measure",
        "    critique --> rewrite & measure",
        "    critique ==> rewrite",
        "    critique -. подпись .-> rewrite",
        "    critique --- rewrite",
    ],
    ids=["chain", "ampersand", "thick", "dotted-label", "no-arrow"],
)
def test_diagram_unparsed_link(line):
    """Ребро в записи, которую тест не разбирает, не проходит молча"""
    with pytest.raises(AssertionError, match="ребро в неразобранной записи"):
        diagram_edges(f"```mermaid\nflowchart TD\n{line}\n```")


def test_diagram_duplicate_edge():
    """Повтор ребра виден в сверке"""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    doubled = text.replace("    critique --> rewrite\n", "    critique --> rewrite\n" * 2)
    assert diagram_edges(doubled)["critique", "rewrite", False] == 2
