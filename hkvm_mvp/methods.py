from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .retrieval import BM25Index, DenseIndex, build_colbert_backend, build_dense_backend, build_graph, hg_diffusion, passage_scores_from_vertices, personalized_pagerank, seed_vertices
from .schema import ExtractionArtifacts, Hyperedge, QAExample, RetrievalOutput, Triple
from .utils import Timer, top_items


@dataclass
class MethodContext:
    config: dict[str, Any]
    artifacts: ExtractionArtifacts
    seed: int


class BaseMethod:
    name = "base"

    def __init__(self, ctx: MethodContext):
        self.ctx = ctx
        self.top_k = int(ctx.config["retrieval"]["top_k"])

    def retrieve(self, ex: QAExample) -> RetrievalOutput:
        with Timer() as timer:
            scores = self.score(ex)
        ranked = top_items(scores, self.top_k)
        return RetrievalOutput(self.name, ex.id, ranked, {pid: scores[pid] for pid in ranked}, timer.elapsed, self.debug(ex))

    def score(self, ex: QAExample) -> dict[str, float]:
        return {}

    def debug(self, ex: QAExample) -> dict[str, Any]:
        return {}


def _supervision_source(ctx: MethodContext) -> str:
    return str(ctx.config.get("weighting", {}).get("supervision_source", "gold_support"))


def _llm_hyperedge_weight(edge: Hyperedge, weighting_cfg: dict[str, Any]) -> float:
    fact_w = float(weighting_cfg.get("hyperedge_factual_weight", 0.20))
    sal_w = float(weighting_cfg.get("hyperedge_salience_weight", 0.40))
    bridge_w = float(weighting_cfg.get("hyperedge_bridge_weight", 0.40))
    denom = max(1e-9, fact_w + sal_w + bridge_w)
    return (
        fact_w * max(0.0, min(1.0, edge.factual_confidence))
        + sal_w * max(0.0, min(1.0, edge.semantic_salience))
        + bridge_w * max(0.0, min(1.0, edge.bridge_potential))
    ) / denom


class BM25Method(BaseMethod):
    name = "bm25"

    def score(self, ex: QAExample) -> dict[str, float]:
        return BM25Index(ex.passages).search(ex.question, self.top_k)


class DenseMethod(BaseMethod):
    name = "dense"

    def __init__(self, ctx: MethodContext):
        super().__init__(ctx)
        self.backend = build_dense_backend(ctx.config.get("dense", {}), require_neural=False)

    def score(self, ex: QAExample) -> dict[str, float]:
        return self.backend.search(ex.passages, ex.question, self.top_k)

    def debug(self, ex: QAExample) -> dict[str, Any]:
        return self.backend.backend_info()


class DenseFallbackMethod(BaseMethod):
    name = "dense_fallback"

    def score(self, ex: QAExample) -> dict[str, float]:
        return DenseIndex(ex.passages).search(ex.question, self.top_k)

    def debug(self, ex: QAExample) -> dict[str, Any]:
        return {"backend": "lexical_bow_cosine"}


class ContrieverMethod(BaseMethod):
    name = "contriever"

    def __init__(self, ctx: MethodContext):
        super().__init__(ctx)
        self.backend = build_dense_backend(ctx.config.get("dense", {}), require_neural=True)

    def score(self, ex: QAExample) -> dict[str, float]:
        return self.backend.search(ex.passages, ex.question, self.top_k)

    def debug(self, ex: QAExample) -> dict[str, Any]:
        return self.backend.backend_info()


class ColBERTv2Method(BaseMethod):
    name = "colbertv2"

    def __init__(self, ctx: MethodContext):
        super().__init__(ctx)
        self.backend = build_colbert_backend(ctx.config.get("colbertv2", {}))

    def score(self, ex: QAExample) -> dict[str, float]:
        return self.backend.search(ex.passages, ex.question, self.top_k)

    def debug(self, ex: QAExample) -> dict[str, Any]:
        return self.backend.backend_info()


class KGPPRMethod(BaseMethod):
    name = "kg_ppr"
    weighted = False

    def score(self, ex: QAExample) -> dict[str, float]:
        triples = self.ctx.artifacts.triples_by_example.get(ex.id, [])
        vertices = {v for t in triples for v in t.vertices()}
        seeds = seed_vertices(ex.question, vertices)
        if not seeds:
            return BM25Index(ex.passages).search(ex.question, self.top_k)
        cfg = self.ctx.config["retrieval"]
        boost = float(self.ctx.config["weighting"].get("supervised_boost", 2.0))
        source = _supervision_source(self.ctx)
        gold = ex.gold_passage_ids if source == "gold_support" else set()
        graph = build_graph(
            triples,
            weighted=self.weighted,
            gold_passage_ids=gold,
            boost=boost,
            weight_source=source,
            llm_weighting=self.ctx.config.get("weighting", {}),
        )
        vertex_scores = personalized_pagerank(
            graph,
            seeds,
            alpha=float(cfg.get("ppr_alpha", 0.5)),
            iterations=int(cfg.get("ppr_iterations", 40)),
            tolerance=float(cfg.get("ppr_tolerance", 1e-7)),
        )
        return passage_scores_from_vertices(vertex_scores, triples)


class WeightedKGPPRMethod(KGPPRMethod):
    name = "weighted_kg_ppr"
    weighted = True


class HyperRAGMethod(BaseMethod):
    name = "hyper_rag"

    def score(self, ex: QAExample) -> dict[str, float]:
        triples = self.ctx.artifacts.triples_by_example.get(ex.id, [])
        hyperedges = self.ctx.artifacts.cooccurrence_hyperedges_by_example.get(ex.id, [])
        vertices = {v for t in triples for v in t.vertices()}
        seeds = seed_vertices(ex.question, vertices)
        if not seeds:
            return BM25Index(ex.passages).search(ex.question, self.top_k)
        cfg = self.ctx.config["retrieval"]
        return hg_diffusion(hyperedges, triples, seeds, float(cfg.get("hg_lambda", 0.35)), int(cfg.get("hg_steps", 1)))


def _select_hyperedges(ctx: MethodContext, ex: QAExample, weighted: bool = False) -> list[Hyperedge]:
    ab = ctx.config.get("ablations", {})
    if ab.get("random_hyperedges"):
        edges = list(ctx.artifacts.random_hyperedges_by_example.get(ex.id, []))
    elif ab.get("cooccurrence_hg") and not ab.get("answer_path_hg"):
        edges = list(ctx.artifacts.cooccurrence_hyperedges_by_example.get(ex.id, []))
    else:
        edges = list(ctx.artifacts.hyperedges_by_example.get(ex.id, []))
        if ab.get("all_hyperedges", True):
            edges.extend(ctx.artifacts.cooccurrence_hyperedges_by_example.get(ex.id, []))
    if ab.get("low_order_only"):
        edges = [e for e in edges if len(e.vertices) <= 2]
    if ab.get("high_order_only"):
        edges = [e for e in edges if len(e.vertices) >= 3]
    if ab.get("gold_hyperedges"):
        gold = ex.gold_passage_ids
        edges = [e for e in edges if set(e.passage_ids) & gold]
    if weighted:
        source = _supervision_source(ctx)
        weighting_cfg = ctx.config["weighting"]
        boost = float(weighting_cfg.get("supervised_boost", 2.0))
        out: list[Hyperedge] = []
        for edge in edges:
            copied = Hyperedge(**edge.__dict__)
            if source == "gold_support" and set(copied.passage_ids) & ex.gold_passage_ids:
                copied.weight *= boost
            elif source == "llm_confidence":
                copied.weight *= 1.0 + max(0.0, boost - 1.0) * _llm_hyperedge_weight(copied, weighting_cfg)
            out.append(copied)
        return out
    return edges


class StaticHGMethod(BaseMethod):
    name = "static_hg"
    weighted = False

    def score(self, ex: QAExample) -> dict[str, float]:
        triples = self.ctx.artifacts.triples_by_example.get(ex.id, [])
        hyperedges = _select_hyperedges(self.ctx, ex, weighted=self.weighted)
        vertices = {v for t in triples for v in t.vertices()}
        seeds = seed_vertices(ex.question, vertices)
        if not seeds:
            return BM25Index(ex.passages).search(ex.question, self.top_k)
        cfg = self.ctx.config["retrieval"]
        steps = 2 if self.ctx.config.get("ablations", {}).get("diffusion_steps_2") else int(cfg.get("hg_steps", 1))
        return hg_diffusion(hyperedges, triples, seeds, float(cfg.get("hg_lambda", 0.35)), steps)


class WeightedHGKVMethod(StaticHGMethod):
    name = "weighted_hg_kv"
    weighted = True

    def __init__(self, ctx: MethodContext):
        super().__init__(ctx)
        self.learned_weights = self._train_weights()

    def _train_weights(self) -> dict[str, float]:
        if _supervision_source(self.ctx) != "gold_support":
            return {}
        weights: dict[str, float] = {}
        lr = float(self.ctx.config["weighting"].get("learning_rate", 0.08))
        epochs = int(self.ctx.config["weighting"].get("contrastive_epochs", 5))
        examples_seen = list(self.ctx.artifacts.hyperedges_by_example.keys())
        rng = random.Random(self.ctx.seed)
        for _ in range(epochs):
            rng.shuffle(examples_seen)
            for ex_id in examples_seen:
                edges = self.ctx.artifacts.hyperedges_by_example.get(ex_id, [])
                gold = self.ctx.artifacts.gold_support_by_example.get(ex_id, set())
                pos = [e for e in edges if set(e.passage_ids) & gold]
                neg = [e for e in edges if not (set(e.passage_ids) & gold)]
                for edge in pos:
                    weights[edge.id] = weights.get(edge.id, 1.0) + lr
                for edge in neg[: int(self.ctx.config["weighting"].get("negative_samples", 12))]:
                    weights[edge.id] = max(0.1, weights.get(edge.id, 1.0) - lr / 2.0)
        return weights

    def score(self, ex: QAExample) -> dict[str, float]:
        triples = self.ctx.artifacts.triples_by_example.get(ex.id, [])
        hyperedges = _select_hyperedges(self.ctx, ex, weighted=True)
        for edge in hyperedges:
            edge.weight *= self.learned_weights.get(edge.id, 1.0)
        vertices = {v for t in triples for v in t.vertices()}
        seeds = seed_vertices(ex.question, vertices)
        if not seeds:
            return BM25Index(ex.passages).search(ex.question, self.top_k)
        cfg = self.ctx.config["retrieval"]
        steps = 2 if self.ctx.config.get("ablations", {}).get("diffusion_steps_2") else int(cfg.get("hg_steps", 1))
        return hg_diffusion(hyperedges, triples, seeds, float(cfg.get("hg_lambda", 0.35)), steps)


METHODS = {
    "bm25": BM25Method,
    "dense": DenseMethod,
    "dense_fallback": DenseFallbackMethod,
    "contriever": ContrieverMethod,
    "colbertv2": ColBERTv2Method,
    "kg_ppr": KGPPRMethod,
    "hyper_rag": HyperRAGMethod,
    "static_hg": StaticHGMethod,
    "weighted_kg_ppr": WeightedKGPPRMethod,
    "weighted_hg_kv": WeightedHGKVMethod,
}
