from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Passage:
    id: str
    title: str
    text: str

    def full_text(self) -> str:
        return f"{self.title}\n{self.text}".strip()


@dataclass
class QAExample:
    id: str
    question: str
    answers: list[str]
    passages: list[Passage]
    gold_passage_ids: set[str]
    supporting_facts: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Triple:
    head: str
    relation: str
    tail: str
    passage_id: str
    confidence: float = 1.0
    source: str = "heuristic"
    factual_confidence: float = 1.0
    semantic_salience: float = 0.5
    bridge_potential: float = 0.0

    def vertices(self) -> tuple[str, str]:
        return (self.head, self.tail)


@dataclass
class Hyperedge:
    id: str
    vertices: tuple[str, ...]
    passage_ids: tuple[str, ...]
    relation: str
    bridge: str = ""
    weight: float = 1.0
    confidence: float = 1.0
    source: str = "answer_path"
    factual_confidence: float = 1.0
    semantic_salience: float = 0.5
    bridge_potential: float = 0.0


@dataclass
class ExtractionArtifacts:
    triples_by_example: dict[str, list[Triple]]
    hyperedges_by_example: dict[str, list[Hyperedge]]
    cooccurrence_hyperedges_by_example: dict[str, list[Hyperedge]]
    random_hyperedges_by_example: dict[str, list[Hyperedge]]
    gold_support_by_example: dict[str, set[str]]


@dataclass
class RetrievalOutput:
    method: str
    example_id: str
    ranked_passage_ids: list[str]
    scores: dict[str, float]
    latency_seconds: float
    debug: dict[str, Any] = field(default_factory=dict)


def to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, tuple):
        return list(obj)
    return obj
