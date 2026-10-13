"""Схема графа в README совпадает с графом в коде"""

import re
from pathlib import Path

import pytest
from langgraph.graph import END, START

from ruts_adapt.graph import build_graph

ROOT = Path(__file__).parent.parent
# Ребро схемы: узел, необязательная фигура с подписью, стрелка, необязательная подпись ребра, узел
EDGE = re.compile(r"^\s*(\w+)[^\n]*?\s(-->|-\.->)(?:\|[^|\n]*\|)?\s*(\w+)", re.MULTILINE)
ENDS = {"start": START, "done": END}


def diagram_edges(path: Path) -> set[tuple[str, str, bool]]:
    """Ребра блока mermaid: откуда, куда, условное ли (пунктир)"""
    (block,) = re.findall(r"```mermaid\n(.*?)```", path.read_text(encoding="utf-8"), re.DOTALL)
    return {
        (ENDS.get(source, source), ENDS.get(target, target), arrow == "-.->")
        for source, arrow, target in EDGE.findall(block)
    }


@pytest.mark.parametrize("name", ["README.md", "README.en.md"])
def test_diagram_matches_graph(name):
    graph = build_graph().get_graph()
    expected = {(edge.source, edge.target, edge.conditional) for edge in graph.edges}
    assert diagram_edges(ROOT / name) == expected
