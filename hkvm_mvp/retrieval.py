from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable

from .extraction import extract_entities
from .schema import Hyperedge, Passage, Triple
from .utils import bow, cosine_dict, tokenize, top_items


class BM25Index:
    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(p.full_text()) for p in passages]
        self.doc_len = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        df: Counter[str] = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
        self.idf = {t: math.log(1 + (len(passages) - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(toks) for toks in self.doc_tokens]

    def search(self, query: str, top_k: int) -> dict[str, float]:
        q = tokenize(query)
        scores: dict[str, float] = {}
        for i, passage in enumerate(self.passages):
            score = 0.0
            dl = self.doc_len[i] or 1
            for term in q:
                freq = self.tf[i].get(term, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl))
                score += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / denom
            if score > 0:
                scores[passage.id] = score
        return {pid: scores[pid] for pid in top_items(scores, top_k)}


class DenseIndex:
    def __init__(self, passages: list[Passage]):
        self.passages = passages
        self.doc_vecs = [bow(p.full_text()) for p in passages]

    def search(self, query: str, top_k: int) -> dict[str, float]:
        qv = bow(query)
        scores = {p.id: cosine_dict(qv, dv) for p, dv in zip(self.passages, self.doc_vecs)}
        scores = {k: v for k, v in scores.items() if v > 0}
        return {pid: scores[pid] for pid in top_items(scores, top_k)}

    def backend_info(self) -> dict[str, Any]:
        return {"backend": "lexical_bow_cosine"}


class LexicalDenseBackend:
    def search(self, passages: list[Passage], query: str, top_k: int) -> dict[str, float]:
        return DenseIndex(passages).search(query, top_k)

    def backend_info(self) -> dict[str, Any]:
        return {"backend": "lexical_bow_cosine"}


class ContrieverIndex:
    _MODEL_CACHE: dict[tuple[str, str | None, str, bool], tuple[Any, Any, Any, str]] = {}

    def __init__(
        self,
        model_name: str = "facebook/contriever",
        batch_size: int = 32,
        device: str = "auto",
        max_length: int = 512,
        normalize: bool = True,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.max_length = int(max_length)
        self.normalize = bool(normalize)
        self.cache_dir = cache_dir
        self.local_files_only = bool(local_files_only)
        self.torch, self.tokenizer, self.model, self.device = self._load_model(model_name, cache_dir, device, self.local_files_only)

    @classmethod
    def _resolve_device(cls, torch: Any, requested: str) -> str:
        if requested and requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @classmethod
    def _load_model(
        cls,
        model_name: str,
        cache_dir: str | None,
        requested_device: str,
        local_files_only: bool,
    ) -> tuple[Any, Any, Any, str]:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Contriever retrieval requires `torch` and `transformers` in the active Python environment.") from exc
        device = cls._resolve_device(torch, requested_device)
        key = (model_name, cache_dir, device, local_files_only)
        if key in cls._MODEL_CACHE:
            return cls._MODEL_CACHE[key]
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_files_only)
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_files_only)
        model.to(device)
        model.eval()
        cls._MODEL_CACHE[key] = (torch, tokenizer, model, device)
        return cls._MODEL_CACHE[key]

    def _encode(self, texts: list[str]):
        if not texts:
            raise ValueError("ContrieverIndex._encode received no texts.")
        chunks = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                batch = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                hidden = outputs.last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                if self.normalize:
                    embeddings = self.torch.nn.functional.normalize(embeddings, p=2, dim=1)
                chunks.append(embeddings.detach().cpu())
        return self.torch.cat(chunks, dim=0)

    def search(self, passages: list[Passage], query: str, top_k: int) -> dict[str, float]:
        if not passages:
            return {}
        passage_texts = [p.full_text() for p in passages]
        query_embedding = self._encode([query])[0]
        passage_embeddings = self._encode(passage_texts)
        scores_tensor = passage_embeddings @ query_embedding
        scores = {p.id: float(score) for p, score in zip(passages, scores_tensor.tolist())}
        return {pid: scores[pid] for pid in top_items(scores, top_k)}

    def backend_info(self) -> dict[str, Any]:
        return {
            "backend": "contriever",
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "normalize": self.normalize,
            "cache_dir": self.cache_dir,
        }


def build_dense_backend(config: dict[str, Any] | None, require_neural: bool = False):
    cfg = config or {}
    backend = str(cfg.get("backend", "lexical")).lower()
    if require_neural:
        backend = "contriever"
    if backend in {"contriever", "facebook/contriever", "neural"}:
        return ContrieverIndex(
            model_name=str(cfg.get("model_name", "facebook/contriever")),
            batch_size=int(cfg.get("batch_size", 32)),
            device=str(cfg.get("device", "auto")),
            max_length=int(cfg.get("max_length", 512)),
            normalize=bool(cfg.get("normalize", True)),
            cache_dir=cfg.get("cache_dir"),
            local_files_only=bool(cfg.get("local_files_only", False)),
        )
    if backend == "auto":
        try:
            return build_dense_backend({**cfg, "backend": "contriever"}, require_neural=False)
        except Exception:
            if not bool(cfg.get("fallback_on_error", True)):
                raise
            return LexicalDenseBackend()
    if backend in {"lexical", "fallback", "bow", "dense_fallback"}:
        return LexicalDenseBackend()
    raise ValueError(f"Unsupported dense backend: {backend}")


class ColBERTv2Index:
    _MODEL_CACHE: dict[tuple[str, str | None, str, bool], tuple[Any, Any, Any, Any, str]] = {}

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        batch_size: int = 16,
        device: str = "auto",
        query_max_length: int = 32,
        doc_max_length: int = 180,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.query_max_length = int(query_max_length)
        self.doc_max_length = int(doc_max_length)
        self.cache_dir = cache_dir
        self.local_files_only = bool(local_files_only)
        self.torch, self.tokenizer, self.model, self.projection, self.device = self._load_model(model_name, cache_dir, device, self.local_files_only)

    @classmethod
    def _load_model(
        cls,
        model_name: str,
        cache_dir: str | None,
        requested_device: str,
        local_files_only: bool,
    ) -> tuple[Any, Any, Any, Any, str]:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("ColBERTv2 retrieval requires `torch` and `transformers` in the active Python environment.") from exc
        device = ContrieverIndex._resolve_device(torch, requested_device)
        key = (model_name, cache_dir, device, local_files_only)
        if key in cls._MODEL_CACHE:
            return cls._MODEL_CACHE[key]
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_files_only)
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_files_only)
        model.to(device)
        model.eval()
        projection = cls._load_projection(torch, model_name, cache_dir, local_files_only, device)
        cls._MODEL_CACHE[key] = (torch, tokenizer, model, projection, device)
        return cls._MODEL_CACHE[key]

    @staticmethod
    def _load_projection(torch: Any, model_name: str, cache_dir: str | None, local_files_only: bool, device: str) -> Any | None:
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file as load_safetensors
        except ImportError:
            return None
        state: dict[str, Any] = {}
        try:
            weights_path = hf_hub_download(
                repo_id=model_name,
                filename="model.safetensors",
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            state = load_safetensors(weights_path, device="cpu")
        except Exception:
            try:
                weights_path = hf_hub_download(
                    repo_id=model_name,
                    filename="pytorch_model.bin",
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
                state = torch.load(weights_path, map_location="cpu")
            except Exception:
                return None
        weight = state.get("linear.weight")
        if weight is None:
            return None
        projection = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        projection.weight.data.copy_(weight)
        projection.to(device)
        projection.eval()
        return projection

    def _encode_tokens(self, texts: list[str], max_length: int) -> list[tuple[Any, Any]]:
        encoded: list[tuple[Any, Any]] = []
        if not texts:
            return encoded
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                batch = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                hidden = outputs.last_hidden_state
                if self.projection is not None:
                    hidden = self.projection(hidden)
                hidden = self.torch.nn.functional.normalize(hidden, p=2, dim=2)
                mask = batch["attention_mask"].bool()
                for idx in range(hidden.shape[0]):
                    token_embeddings = hidden[idx][mask[idx]].detach().cpu()
                    token_mask = mask[idx][mask[idx]].detach().cpu()
                    encoded.append((token_embeddings, token_mask))
        return encoded

    def search(self, passages: list[Passage], query: str, top_k: int) -> dict[str, float]:
        if not passages:
            return {}
        query_tokens = self._encode_tokens([query], self.query_max_length)[0][0]
        passage_texts = [p.full_text() for p in passages]
        doc_tokens = self._encode_tokens(passage_texts, self.doc_max_length)
        scores: dict[str, float] = {}
        for passage, (doc_embedding, _doc_mask) in zip(passages, doc_tokens):
            if query_tokens.numel() == 0 or doc_embedding.numel() == 0:
                continue
            sim = query_tokens @ doc_embedding.T
            score = float(sim.max(dim=1).values.sum().item())
            scores[passage.id] = score
        return {pid: scores[pid] for pid in top_items(scores, top_k)}

    def backend_info(self) -> dict[str, Any]:
        return {
            "backend": "colbertv2_late_interaction",
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "query_max_length": self.query_max_length,
            "doc_max_length": self.doc_max_length,
            "cache_dir": self.cache_dir,
            "projection": self.projection is not None,
        }


def build_colbert_backend(config: dict[str, Any] | None):
    cfg = config or {}
    return ColBERTv2Index(
        model_name=str(cfg.get("model_name", "colbert-ir/colbertv2.0")),
        batch_size=int(cfg.get("batch_size", 16)),
        device=str(cfg.get("device", "auto")),
        query_max_length=int(cfg.get("query_max_length", 32)),
        doc_max_length=int(cfg.get("doc_max_length", 180)),
        cache_dir=cfg.get("cache_dir"),
        local_files_only=bool(cfg.get("local_files_only", False)),
    )


def build_graph(
    triples: Iterable[Triple],
    weighted: bool = False,
    gold_passage_ids: set[str] | None = None,
    boost: float = 2.0,
    weight_source: str = "gold_support",
    llm_weighting: dict[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    graph: dict[str, dict[str, float]] = defaultdict(dict)
    gold_passage_ids = gold_passage_ids or set()
    llm_weighting = llm_weighting or {}
    fact_w = float(llm_weighting.get("triple_factual_weight", 0.25))
    sal_w = float(llm_weighting.get("triple_salience_weight", 0.55))
    bridge_w = float(llm_weighting.get("triple_bridge_weight", 0.20))
    denom = max(1e-9, fact_w + sal_w + bridge_w)
    for t in triples:
        h, r = t.head, t.tail
        w = t.confidence
        if weighted:
            if weight_source == "gold_support" and t.passage_id in gold_passage_ids:
                w *= boost
            elif weight_source == "llm_confidence":
                score = (
                    fact_w * max(0.0, min(1.0, t.factual_confidence))
                    + sal_w * max(0.0, min(1.0, t.semantic_salience))
                    + bridge_w * max(0.0, min(1.0, t.bridge_potential))
                ) / denom
                w *= 1.0 + max(0.0, boost - 1.0) * score
        graph[h][r] = graph[h].get(r, 0.0) + w
        graph[r][h] = graph[r].get(h, 0.0) + w
    return graph


def personalized_pagerank(graph: dict[str, dict[str, float]], seeds: dict[str, float], alpha: float, iterations: int, tolerance: float) -> dict[str, float]:
    if not graph or not seeds:
        return {}
    nodes = set(graph.keys())
    for nbrs in graph.values():
        nodes.update(nbrs.keys())
    seed_sum = sum(seeds.values()) or 1.0
    p0 = {n: seeds.get(n, 0.0) / seed_sum for n in nodes}
    scores = dict(p0)
    for _ in range(iterations):
        new_scores = {n: alpha * p0.get(n, 0.0) for n in nodes}
        for node, nbrs in graph.items():
            total = sum(nbrs.values())
            if total <= 0:
                continue
            share = (1.0 - alpha) * scores.get(node, 0.0)
            for nbr, weight in nbrs.items():
                new_scores[nbr] = new_scores.get(nbr, 0.0) + share * weight / total
        delta = sum(abs(new_scores.get(n, 0.0) - scores.get(n, 0.0)) for n in nodes)
        scores = new_scores
        if delta < tolerance:
            break
    return scores


def seed_vertices(question: str, vertices: Iterable[str]) -> dict[str, float]:
    q_tokens = set(tokenize(question))
    q_entities = {e.lower() for e in extract_entities(question, max_entities=16)}
    seeds: dict[str, float] = {}
    for vertex in vertices:
        vtoks = set(tokenize(vertex))
        if vertex.lower() in q_entities:
            seeds[vertex] = 1.0
        else:
            overlap = len(q_tokens & vtoks)
            if overlap:
                seeds[vertex] = float(overlap) / max(1, len(vtoks))
    return seeds


def passage_scores_from_vertices(vertex_scores: dict[str, float], triples: Iterable[Triple]) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for triple in triples:
        score = vertex_scores.get(triple.head, 0.0) + vertex_scores.get(triple.tail, 0.0)
        if score:
            scores[triple.passage_id] += score * triple.confidence
    return dict(scores)


def hg_diffusion(
    hyperedges: list[Hyperedge],
    triples: list[Triple],
    seeds: dict[str, float],
    lam: float,
    steps: int,
) -> dict[str, float]:
    if not hyperedges or not seeds:
        return {}
    vertex_scores = dict(seeds)
    vertex_degree: Counter[str] = Counter()
    for edge in hyperedges:
        for v in edge.vertices:
            vertex_degree[v] += 1
    for _ in range(max(1, steps)):
        edge_scores: dict[str, float] = {}
        for edge in hyperedges:
            denom = max(1, len(edge.vertices))
            edge_scores[edge.id] = edge.weight * sum(vertex_scores.get(v, 0.0) for v in edge.vertices) / denom
        new_vertex_scores: dict[str, float] = defaultdict(float)
        for edge in hyperedges:
            es = edge_scores.get(edge.id, 0.0)
            if es == 0:
                continue
            for vertex in edge.vertices:
                new_vertex_scores[vertex] += es / max(1, vertex_degree.get(vertex, 1))
        all_vertices = set(vertex_scores) | set(new_vertex_scores)
        vertex_scores = {v: lam * seeds.get(v, 0.0) + (1.0 - lam) * new_vertex_scores.get(v, 0.0) for v in all_vertices}
    passage_scores = passage_scores_from_vertices(vertex_scores, triples)
    for edge in hyperedges:
        es = sum(vertex_scores.get(v, 0.0) for v in edge.vertices) * edge.weight / max(1, len(edge.vertices))
        for pid in edge.passage_ids:
            passage_scores[pid] = passage_scores.get(pid, 0.0) + es
    return passage_scores
