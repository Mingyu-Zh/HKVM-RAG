from __future__ import annotations

import hashlib
import itertools
import random
import re
from pathlib import Path
from typing import Any

from .reproducibility import cache_context, validate_cache_manifest, write_cache_manifest
from .schema import ExtractionArtifacts, Hyperedge, QAExample, Triple
from .utils import ensure_dir, read_json, tokenize, write_json

ENTITY_RE = re.compile(r"(?:[A-Z][A-Za-z0-9_\-]+(?:\s+[A-Z][A-Za-z0-9_\-]+){0,4})")


def canonical_entity(text: str) -> str:
    return " ".join(str(text).strip().split())


def extract_entities(text: str, max_entities: int = 12, min_len: int = 2) -> list[str]:
    candidates: list[str] = []
    for match in ENTITY_RE.finditer(text or ""):
        ent = canonical_entity(match.group(0))
        if len(ent) >= min_len and ent.lower() not in {"the", "a", "an"}:
            candidates.append(ent)
    if not candidates:
        toks = [t for t in tokenize(text) if len(t) >= max(min_len, 4)]
        candidates.extend(toks[:max_entities])
    seen: set[str] = set()
    out: list[str] = []
    for ent in candidates:
        key = ent.lower()
        if key not in seen:
            seen.add(key)
            out.append(ent)
        if len(out) >= max_entities:
            break
    return out


def _triple_cache_path(cache_dir: str | Path, dataset_name: str, seed: int, cache_hash: str | None = None) -> Path:
    if cache_hash:
        return Path(cache_dir) / f"{dataset_name}.seed{seed}.{cache_hash}.triples.json"
    return Path(cache_dir) / f"{dataset_name}.seed{seed}.triples.json"


def _serialize_triples(data: dict[str, list[Triple]]) -> dict[str, list[dict[str, Any]]]:
    return {
        ex_id: [
            {
                "head": t.head,
                "relation": t.relation,
                "tail": t.tail,
                "passage_id": t.passage_id,
                "confidence": t.confidence,
                "source": t.source,
                "factual_confidence": t.factual_confidence,
                "semantic_salience": t.semantic_salience,
                "bridge_potential": t.bridge_potential,
            }
            for t in triples
        ]
        for ex_id, triples in data.items()
    }


def _deserialize_triples(data: dict[str, list[dict[str, Any]]]) -> dict[str, list[Triple]]:
    return {ex_id: [Triple(**t) for t in triples] for ex_id, triples in data.items()}


class OpenIEExtractor:
    def __init__(self, cfg: dict[str, Any], dataset_name: str, seed: int, cache_ctx: dict[str, Any] | None = None):
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.seed = seed
        self.cache_ctx = cache_ctx or {}

    def extract(self, examples: list[QAExample]) -> dict[str, list[Triple]]:
        cache_dir = self.cfg.get("cache_dir")
        cache_hash = self.cache_ctx.get("cache_hash")
        cache_path = _triple_cache_path(cache_dir, self.dataset_name, self.seed, cache_hash) if cache_dir else None
        if cache_path and cache_path.exists():
            if self.cache_ctx:
                validate_cache_manifest(cache_path, self.cache_ctx)
            return _deserialize_triples(read_json(cache_path))
        triples = {ex.id: self._extract_example(ex) for ex in examples}
        if cache_path:
            ensure_dir(cache_path.parent)
            write_json(cache_path, _serialize_triples(triples))
            if self.cache_ctx:
                write_cache_manifest(
                    cache_path,
                    self.cache_ctx,
                    {
                        "examples": len(examples),
                        "triples": sum(len(v) for v in triples.values()),
                    },
                )
        return triples

    def _extract_example(self, ex: QAExample) -> list[Triple]:
        max_entities = int(self.cfg.get("max_entities_per_passage", 12))
        min_len = int(self.cfg.get("min_entity_len", 2))
        triples: list[Triple] = []
        answer_terms = {a.lower() for a in ex.answers if a}
        question_entities = set(extract_entities(ex.question, max_entities=max_entities, min_len=min_len))
        for passage in ex.passages:
            text = passage.full_text()
            entities = extract_entities(text, max_entities=max_entities, min_len=min_len)
            for answer in answer_terms:
                if answer and answer in text.lower():
                    entities.append(answer)
            entities = list(dict.fromkeys([canonical_entity(x) for x in entities if x]))
            for ent in entities:
                if ent in question_entities:
                    triples.append(Triple(ent, "mentioned_in_question_context", passage.id, passage.id, 1.0, "heuristic"))
            for head, tail in itertools.combinations(entities[:max_entities], 2):
                relation = "co_occurs"
                confidence = 0.65
                if passage.id in ex.gold_passage_ids:
                    relation = "support_candidate"
                    confidence = 0.85
                triples.append(Triple(head, relation, tail, passage.id, confidence, "heuristic"))
        return triples


def gold_support_annotations(examples: list[QAExample]) -> dict[str, set[str]]:
    return {ex.id: set(ex.gold_passage_ids) for ex in examples}


def build_answer_path_hyperedges(ex: QAExample, triples: list[Triple], use_gold_support: bool = True) -> list[Hyperedge]:
    by_bridge: dict[str, list[Triple]] = {}
    gold_ids = set(ex.gold_passage_ids) if use_gold_support else set()
    for triple in triples:
        if gold_ids and triple.passage_id not in gold_ids:
            continue
        for bridge in triple.vertices():
            by_bridge.setdefault(bridge.lower(), []).append(triple)
    hyperedges: list[Hyperedge] = []
    for bridge_key, group in by_bridge.items():
        vertices = sorted({v for t in group for v in t.vertices()})
        passages = sorted({t.passage_id for t in group})
        if len(vertices) >= 3 and len(group) >= 2:
            raw_id = f"{ex.id}|answer_path|{bridge_key}|{'|'.join(vertices)}|{'|'.join(passages)}"
            hid = hashlib.md5(raw_id.encode("utf-8")).hexdigest()[:16]
            hyperedges.append(
                Hyperedge(
                    id=f"ap_{hid}",
                    vertices=tuple(vertices),
                    passage_ids=tuple(passages),
                    relation="answer_path",
                    bridge=bridge_key,
                    weight=1.0,
                    confidence=min(1.0, sum(t.confidence for t in group) / max(1, len(group))),
                    source="answer_path",
                    factual_confidence=min(1.0, sum(t.factual_confidence for t in group) / max(1, len(group))),
                    semantic_salience=min(1.0, sum(t.semantic_salience for t in group) / max(1, len(group))),
                    bridge_potential=min(1.0, max([t.bridge_potential for t in group] or [0.0])),
                )
            )
    return hyperedges


def build_cooccurrence_hyperedges(ex: QAExample, triples: list[Triple]) -> list[Hyperedge]:
    by_passage: dict[str, set[str]] = {}
    for triple in triples:
        by_passage.setdefault(triple.passage_id, set()).update(triple.vertices())
    out: list[Hyperedge] = []
    for pid, vertices in by_passage.items():
        verts = tuple(sorted(vertices))
        if len(verts) >= 3:
            hid = hashlib.md5(f"{ex.id}|co|{pid}|{verts}".encode("utf-8")).hexdigest()[:16]
            out.append(Hyperedge(f"co_{hid}", verts, (pid,), "cooccurrence", "", 1.0, 0.55, "cooccurrence"))
    return out


def build_random_hyperedges(ex: QAExample, triples: list[Triple], count: int, size: int, seed: int) -> list[Hyperedge]:
    stable_ex_id = int(hashlib.md5(ex.id.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed + stable_ex_id % 100000)
    vertices = sorted({v for t in triples for v in t.vertices()})
    passage_ids = [p.id for p in ex.passages]
    if len(vertices) < size or not passage_ids:
        return []
    out: list[Hyperedge] = []
    for i in range(count):
        verts = tuple(sorted(rng.sample(vertices, size)))
        pid = rng.choice(passage_ids)
        hid = hashlib.md5(f"{ex.id}|rand|{i}|{verts}|{pid}".encode("utf-8")).hexdigest()[:16]
        out.append(Hyperedge(f"rand_{hid}", verts, (pid,), "random", "", 1.0, 0.1, "random"))
    return out


def build_artifacts(
    examples: list[QAExample],
    dataset_name: str,
    cfg: dict[str, Any],
    seed: int,
    dataset_cfg: dict[str, Any] | None = None,
) -> ExtractionArtifacts:
    cache_ctx = cache_context(cfg, dataset_name, dataset_cfg or {}, seed)
    openie_cfg = cfg.get("openie", {})
    backend = str(openie_cfg.get("backend", "heuristic")).lower()
    llm_hyperedges_by_example: dict[str, list[Hyperedge]] = {}
    if backend in {"deepseek", "deepseek_v4", "deepseek-v4", "deepseek_v4_flash", "deepseek-v4-flash"}:
        from .llm_openie import DeepSeekV4OpenIEExtractor

        triples_by_example, llm_hyperedges_by_example = DeepSeekV4OpenIEExtractor(openie_cfg, dataset_name, seed, cache_ctx).extract(examples)
    else:
        triples_by_example = OpenIEExtractor(openie_cfg, dataset_name, seed, cache_ctx).extract(examples)
    gold = gold_support_annotations(examples)
    retrieval_cfg = cfg.get("retrieval", {})
    use_gold_answer_path = bool(openie_cfg.get("use_gold_answer_path", backend == "heuristic"))
    use_llm_hyperedges = bool(openie_cfg.get("use_llm_hyperedges", backend != "heuristic"))
    hyperedges_by_example: dict[str, list[Hyperedge]] = {}
    co_by_example: dict[str, list[Hyperedge]] = {}
    rand_by_example: dict[str, list[Hyperedge]] = {}
    for ex in examples:
        triples = triples_by_example.get(ex.id, [])
        answer_path_edges = build_answer_path_hyperedges(ex, triples, use_gold_support=use_gold_answer_path)
        if use_llm_hyperedges:
            seen = {edge.id for edge in answer_path_edges}
            merged = list(answer_path_edges)
            for edge in llm_hyperedges_by_example.get(ex.id, []):
                if edge.id not in seen:
                    seen.add(edge.id)
                    merged.append(edge)
            hyperedges_by_example[ex.id] = merged
        else:
            hyperedges_by_example[ex.id] = answer_path_edges
        co_by_example[ex.id] = build_cooccurrence_hyperedges(ex, triples)
        rand_by_example[ex.id] = build_random_hyperedges(
            ex,
            triples,
            int(retrieval_cfg.get("random_hyperedge_count", 128)),
            int(retrieval_cfg.get("random_hyperedge_size", 3)),
            seed,
        )
    return ExtractionArtifacts(triples_by_example, hyperedges_by_example, co_by_example, rand_by_example, gold)
