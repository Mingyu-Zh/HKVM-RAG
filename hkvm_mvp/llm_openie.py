from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from threading import Lock
from typing import Any

from .reproducibility import validate_cache_manifest, write_cache_manifest
from .schema import Hyperedge, Passage, QAExample, Triple
from .utils import ensure_dir, read_json, write_json


def _clamp_confidence(value: Any, default: float = 0.7) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def _field_score(item: dict[str, Any], key: str, fallback: str = "confidence", default: float = 0.5) -> float:
    if key in item:
        return _clamp_confidence(item.get(key), default)
    if fallback in item:
        return _clamp_confidence(item.get(fallback), default)
    return _clamp_confidence(default, default)


def _canonical_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.md5(raw.encode('utf-8')).hexdigest()[:16]}"


def _load_env_file(path: str | None) -> None:
    if not path:
        return
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _llm_cache_path(cache_dir: str | Path, dataset_name: str, seed: int, cache_hash: str | None) -> Path:
    suffix = f".{cache_hash}" if cache_hash else ""
    return Path(cache_dir) / f"{dataset_name}.seed{seed}{suffix}.llm_openie.json"


def _indexed_cache_path(cfg: dict[str, Any], dataset_name: str) -> Path | None:
    index_path = cfg.get("cache_index_path")
    if not index_path:
        return None
    path = Path(index_path)
    if not path.exists():
        return None
    split = str(cfg.get("cache_split", "") or "")
    data = read_json(path)
    for item in data.get("entries", []):
        if item.get("dataset") != dataset_name:
            continue
        if split and item.get("split") != split:
            continue
        candidate = path.parent / str(item["path"])
        if candidate.exists():
            return candidate
    return None


def _is_content_risk_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "content exists risk" in text or ("invalid_request_error" in text and "content" in text and "risk" in text)


def _serialize_triples(triples: list[Triple]) -> list[dict[str, Any]]:
    return [
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


def _deserialize_triples(items: list[dict[str, Any]]) -> list[Triple]:
    return [Triple(**item) for item in items]


def _serialize_hyperedges(edges: list[Hyperedge]) -> list[dict[str, Any]]:
    return [
        {
            "id": e.id,
            "vertices": list(e.vertices),
            "passage_ids": list(e.passage_ids),
            "relation": e.relation,
            "bridge": e.bridge,
            "weight": e.weight,
            "confidence": e.confidence,
            "source": e.source,
            "factual_confidence": e.factual_confidence,
            "semantic_salience": e.semantic_salience,
            "bridge_potential": e.bridge_potential,
        }
        for e in edges
    ]


def _deserialize_hyperedges(items: list[dict[str, Any]]) -> list[Hyperedge]:
    out: list[Hyperedge] = []
    for item in items:
        copied = dict(item)
        copied["vertices"] = tuple(copied.get("vertices", []))
        copied["passage_ids"] = tuple(copied.get("passage_ids", []))
        out.append(Hyperedge(**copied))
    return out


class DeepSeekV4OpenIEExtractor:
    """Shared no-gold LLM extraction for P2.

    The extractor receives only question/passages allowed by config. It never
    consumes answers, supporting_facts, or gold passage ids.
    """

    def __init__(self, cfg: dict[str, Any], dataset_name: str, seed: int, cache_ctx: dict[str, Any] | None = None):
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.seed = seed
        self.cache_ctx = cache_ctx or {}
        self.model = str(cfg.get("model", "deepseek-v4-flash"))
        self.base_url = str(cfg.get("base_url", "https://api.deepseek.com")).rstrip("/")
        self.timeout = float(cfg.get("timeout_seconds", 120))
        self.max_tokens = int(cfg.get("max_tokens", 4096))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.thinking = str(cfg.get("thinking", "disabled"))
        self.retries = int(cfg.get("retries", 3))
        self.retry_sleep_seconds = float(cfg.get("retry_sleep_seconds", 2.0))
        self.max_passage_chars = int(cfg.get("max_passage_chars", 3500))
        self.include_question = bool(cfg.get("include_question", False))
        self.save_every = max(1, int(cfg.get("save_every", 20)))
        self.request_scope = str(cfg.get("request_scope", "passage")).lower()
        self.schema_version = str(cfg.get("schema_version", "p2_salience_bridge_v1"))
        self.parallel_requests = max(1, int(cfg.get("parallel_requests", 1)))
        self.global_max_workers = max(1, int(cfg.get("global_max_workers", self.parallel_requests)))
        self.max_in_flight = max(self.global_max_workers, int(cfg.get("max_in_flight", self.global_max_workers * 4)))
        self.extract_high_order_relations = bool(cfg.get("extract_high_order_relations", True))
        self.adaptive_backoff = bool(cfg.get("adaptive_backoff", False))
        self.retry_jitter = bool(cfg.get("retry_jitter", False))
        self.max_retry_sleep_seconds = float(cfg.get("max_retry_sleep_seconds", 60.0))
        self.max_triples_per_passage = int(cfg.get("max_triples_per_passage", 5))
        self.max_high_order_per_passage = int(cfg.get("max_high_order_per_passage", 2))
        self.skip_content_risk = bool(cfg.get("skip_content_risk", True))
        self._metrics_lock = Lock()
        self._request_metrics: dict[str, int] = {"ok": 0, "retry": 0, "failed": 0, "skipped_content_risk": 0}

    def extract(self, examples: list[QAExample]) -> tuple[dict[str, list[Triple]], dict[str, list[Hyperedge]]]:
        cache_dir = self.cfg.get("cache_dir")
        cache_seed = 0 if bool(self.cfg.get("seed_invariant_cache", True)) else self.seed
        cache_hash = self.cache_ctx.get("cache_hash")
        cache_path = _llm_cache_path(cache_dir, self.dataset_name, cache_seed, cache_hash) if cache_dir else None
        if cache_path and not cache_path.exists():
            cache_path = _indexed_cache_path(self.cfg, self.dataset_name) or cache_path
        cached = self._read_cache(cache_path)
        triples_by_example: dict[str, list[Triple]] = dict(cached[0])
        hyperedges_by_example: dict[str, list[Hyperedge]] = dict(cached[1])
        missing = [ex for ex in examples if ex.id not in triples_by_example]
        if not missing:
            return triples_by_example, hyperedges_by_example

        api_key = self._api_key()
        if self.request_scope == "passage" and self.global_max_workers > 1:
            return self._extract_missing_passagewise_global(
                missing,
                api_key,
                triples_by_example,
                hyperedges_by_example,
                cache_path,
            )

        completed = 0
        for ex in missing:
            if self.request_scope == "example":
                triples, hyperedges = self._extract_example_with_retries(ex, api_key)
            else:
                triples, hyperedges = self._extract_example_passagewise(ex, api_key)
            triples_by_example[ex.id] = triples
            hyperedges_by_example[ex.id] = hyperedges
            completed += 1
            if cache_path and completed % self.save_every == 0:
                self._write_cache(cache_path, triples_by_example, hyperedges_by_example, complete=False)
            if completed % self.save_every == 0:
                print(f"[hkvm-rag] DeepSeek OpenIE cached {completed}/{len(missing)} missing examples for {self.dataset_name}.")
        if cache_path:
            self._write_cache(cache_path, triples_by_example, hyperedges_by_example, complete=True)
        return triples_by_example, hyperedges_by_example

    def _api_key(self) -> str:
        _load_env_file(self.cfg.get("env_file"))
        direct = str(self.cfg.get("api_key", "") or "").strip()
        if direct and direct not in {"填入你的token", "YOUR_DEEPSEEK_API_KEY", "CHANGE_ME"}:
            return direct
        env_name = str(self.cfg.get("api_key_env", "DEEPSEEK_API_KEY"))
        value = os.environ.get(env_name, "").strip()
        if value in {"填入你的token", "YOUR_DEEPSEEK_API_KEY", "CHANGE_ME"}:
            value = ""
        if not value:
            env_hint = self.cfg.get("env_file") or f"${env_name}"
            raise RuntimeError(
                "DeepSeek API key is missing. "
                f"Set {env_name} in the environment or fill the local secrets file: {env_hint}"
            )
        return value

    def _read_cache(self, cache_path: Path | None) -> tuple[dict[str, list[Triple]], dict[str, list[Hyperedge]]]:
        if not cache_path or not cache_path.exists():
            return {}, {}
        if self.cache_ctx and not self.cfg.get("cache_index_path"):
            validate_cache_manifest(cache_path, self.cache_ctx)
        data = read_json(cache_path)
        examples = data.get("examples", data)
        triples_by_example: dict[str, list[Triple]] = {}
        hyperedges_by_example: dict[str, list[Hyperedge]] = {}
        for ex_id, payload in examples.items():
            triples_by_example[str(ex_id)] = _deserialize_triples(payload.get("triples", []))
            hyperedges_by_example[str(ex_id)] = _deserialize_hyperedges(payload.get("hyperedges", []))
        return triples_by_example, hyperedges_by_example

    def _write_cache(
        self,
        cache_path: Path,
        triples_by_example: dict[str, list[Triple]],
        hyperedges_by_example: dict[str, list[Hyperedge]],
        complete: bool,
    ) -> None:
        ensure_dir(cache_path.parent)
        payload = {
            "metadata": {
                "provider": "deepseek",
                "model": self.model,
                "schema_version": self.schema_version,
                "request_scope": self.request_scope,
                "parallel_requests": self.parallel_requests,
                "global_max_workers": self.global_max_workers,
                "extract_high_order_relations": self.extract_high_order_relations,
                "dataset": self.dataset_name,
                "seed_invariant": bool(self.cfg.get("seed_invariant_cache", True)),
                "complete": complete,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "examples": {
                ex_id: {
                    "triples": _serialize_triples(triples),
                    "hyperedges": _serialize_hyperedges(hyperedges_by_example.get(ex_id, [])),
                }
                for ex_id, triples in triples_by_example.items()
            },
        }
        write_json(cache_path, payload)
        if self.cache_ctx:
            write_cache_manifest(
                cache_path,
                self.cache_ctx,
                {
                    "examples": len(triples_by_example),
                    "triples": sum(len(v) for v in triples_by_example.values()),
                    "llm_hyperedges": sum(len(v) for v in hyperedges_by_example.values()),
                    "complete": complete,
                    "request_metrics": dict(self._request_metrics),
                },
            )

    def _extract_missing_passagewise_global(
        self,
        missing: list[QAExample],
        api_key: str,
        triples_by_example: dict[str, list[Triple]],
        hyperedges_by_example: dict[str, list[Hyperedge]],
        cache_path: Path | None,
    ) -> tuple[dict[str, list[Triple]], dict[str, list[Hyperedge]]]:
        states: dict[str, dict[str, Any]] = {}
        tasks: list[tuple[QAExample, Passage]] = []
        completed_examples = 0
        completed_passages = 0
        for ex in missing:
            states[ex.id] = {"triples": [], "hyperedges": [], "remaining": len(ex.passages)}
            if not ex.passages:
                triples_by_example[ex.id] = []
                hyperedges_by_example[ex.id] = []
                completed_examples += 1
                continue
            for passage in ex.passages:
                tasks.append((ex, passage))
        total_passages = len(tasks)
        task_iter = iter(tasks)

        def submit_next(executor: ThreadPoolExecutor, futures: dict[Future[tuple[list[Triple], list[Hyperedge]]], tuple[QAExample, Passage]]) -> bool:
            try:
                ex, passage = next(task_iter)
            except StopIteration:
                return False
            future = executor.submit(self._extract_passage_with_retries, ex, passage, api_key)
            futures[future] = (ex, passage)
            return True

        with ThreadPoolExecutor(max_workers=self.global_max_workers) as executor:
            futures: dict[Future[tuple[list[Triple], list[Hyperedge]]], tuple[QAExample, Passage]] = {}
            for _ in range(min(self.max_in_flight, total_passages)):
                submit_next(executor, futures)
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    ex, passage = futures.pop(future)
                    try:
                        p_triples, p_hyperedges = future.result()
                    except Exception:
                        with self._metrics_lock:
                            self._request_metrics["failed"] += 1
                        raise
                    state = states[ex.id]
                    state["triples"].extend(p_triples)
                    state["hyperedges"].extend(p_hyperedges)
                    state["remaining"] -= 1
                    completed_passages += 1
                    if state["remaining"] == 0:
                        triples_by_example[ex.id] = self._dedupe_triples(state["triples"])
                        hyperedges_by_example[ex.id] = self._dedupe_hyperedges(state["hyperedges"])
                        completed_examples += 1
                        if cache_path and completed_examples % self.save_every == 0:
                            self._write_cache(cache_path, triples_by_example, hyperedges_by_example, complete=False)
                        if completed_examples % self.save_every == 0:
                            print(
                                f"[hkvm-rag] DeepSeek OpenIE cached {completed_examples}/{len(missing)} missing examples "
                                f"for {self.dataset_name}; passages {completed_passages}/{total_passages}; "
                                f"workers={self.global_max_workers}; request_metrics={self._request_metrics}."
                            )
                    while len(futures) < self.max_in_flight and submit_next(executor, futures):
                        pass
        if cache_path:
            self._write_cache(cache_path, triples_by_example, hyperedges_by_example, complete=True)
        return triples_by_example, hyperedges_by_example

    def _extract_example_with_retries(self, ex: QAExample, api_key: str) -> tuple[list[Triple], list[Hyperedge]]:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                raw = self._call_api(self._example_user_prompt(ex), api_key)
                parsed = self._parse_json(raw)
                return self._to_artifacts(ex, parsed)
            except Exception as exc:  # noqa: BLE001 - retry reports final error with context.
                last_error = exc
                if self.skip_content_risk and _is_content_risk_error(exc):
                    with self._metrics_lock:
                        self._request_metrics["skipped_content_risk"] += 1
                    return [], []
                if attempt < self.retries:
                    time.sleep(self.retry_sleep_seconds * attempt)
        if bool(self.cfg.get("fail_on_error", True)):
            raise RuntimeError(f"DeepSeek OpenIE failed for example {ex.id}: {last_error}") from last_error
        return [], []

    def _extract_example_passagewise(self, ex: QAExample, api_key: str) -> tuple[list[Triple], list[Hyperedge]]:
        triples: list[Triple] = []
        hyperedges: list[Hyperedge] = []
        if self.parallel_requests <= 1 or len(ex.passages) <= 1:
            for passage in ex.passages:
                p_triples, p_hyperedges = self._extract_passage_with_retries(ex, passage, api_key)
                triples.extend(p_triples)
                hyperedges.extend(p_hyperedges)
        else:
            with ThreadPoolExecutor(max_workers=min(self.parallel_requests, len(ex.passages))) as executor:
                futures = {executor.submit(self._extract_passage_with_retries, ex, passage, api_key): passage for passage in ex.passages}
                for future in as_completed(futures):
                    p_triples, p_hyperedges = future.result()
                    triples.extend(p_triples)
                    hyperedges.extend(p_hyperedges)
        return self._dedupe_triples(triples), self._dedupe_hyperedges(hyperedges)

    def _extract_passage_with_retries(self, ex: QAExample, passage: Passage, api_key: str) -> tuple[list[Triple], list[Hyperedge]]:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                raw = self._call_api(self._passage_user_prompt(ex, passage), api_key)
                parsed = self._parse_json(raw)
                with self._metrics_lock:
                    self._request_metrics["ok"] += 1
                return self._to_artifacts(ex, parsed)
            except Exception as exc:  # noqa: BLE001 - retry reports final error with context.
                last_error = exc
                if self.skip_content_risk and _is_content_risk_error(exc):
                    with self._metrics_lock:
                        self._request_metrics["skipped_content_risk"] += 1
                    return [], []
                if attempt < self.retries:
                    with self._metrics_lock:
                        self._request_metrics["retry"] += 1
                    time.sleep(self._retry_sleep_seconds(attempt, exc))
        if bool(self.cfg.get("fail_on_error", True)):
            raise RuntimeError(f"DeepSeek OpenIE failed for example {ex.id}, passage {passage.id}: {last_error}") from last_error
        with self._metrics_lock:
            self._request_metrics["failed"] += 1
        return [], []

    def _retry_sleep_seconds(self, attempt: int, exc: Exception) -> float:
        sleep = self.retry_sleep_seconds * attempt
        if self.adaptive_backoff:
            err = str(exc).lower()
            if "429" in err or "rate" in err or "timeout" in err or "timed out" in err or "overload" in err:
                sleep *= 3.0
        if self.retry_jitter:
            sleep *= random.uniform(0.75, 1.5)
        return min(self.max_retry_sleep_seconds, sleep)

    def _call_api(self, user_prompt: str, api_key: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "thinking": {"type": self.thinking},
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek API HTTP {exc.code}: {detail[:1000]}") from exc
        obj = json.loads(body)
        choices = obj.get("choices") or []
        if not choices:
            raise RuntimeError(f"DeepSeek API returned no choices: {body[:1000]}")
        content = ((choices[0].get("message") or {}).get("content") or "").strip()
        if not content:
            raise RuntimeError(f"DeepSeek API returned empty content: {body[:1000]}")
        return content

    def _system_prompt(self) -> str:
        if not self.extract_high_order_relations:
            return (
                "You are an OpenIE triple extraction engine for RAG indexing. "
                "Return only valid json. Extract compact subject-relation-object triples strictly supported by the supplied passage. "
                "Do not answer the question. Do not infer facts that are not explicitly stated. "
                "Do not use benchmark labels, gold supports, or answers."
            )
        return (
            "You are an OpenIE and high-order relation extraction engine for RAG indexing. "
            "Return only valid json. Extract information strictly supported by the supplied passages. "
            "Do not answer the question. Do not infer facts that are not explicitly stated. "
            "Do not use benchmark labels, gold supports, or answers."
        )

    def _schema_prompt(self) -> str:
        if not self.extract_high_order_relations:
            return (
                "Output compact valid JSON with this exact top-level shape:\n"
                "{\n"
                '  "passages": [\n'
                '    {"passage_id": "string", '
                '"triples": [{"head": "string", "relation": "string", "tail": "string", "factual_confidence": 0.0, "semantic_salience": 0.0, "bridge_potential": 0.0}]}\n'
                "  ]\n"
                "}\n\n"
                "Rules:\n"
                "- Return JSON only; no markdown and no comments.\n"
                "- Use only passage_id values provided below.\n"
                "- Prefer compact canonical entity names.\n"
                f"- Extract at most {self.max_triples_per_passage} salient triples per passage.\n"
                "- Do not return high_order_relations or cross_passage_high_order_relations.\n"
                "- factual_confidence: 0 means weak textual support; 1 means explicitly stated in the passage.\n"
                "- semantic_salience: 0 means peripheral detail; 1 means central relation/event of the passage.\n"
                "- bridge_potential: 0 means unlikely to help multi-hop linking; 1 means connects entities/events/attributes that could bridge to other passages.\n"
                "- Do not set all scores to the same value; use the full 0-1 range when justified.\n"
                "- Omit source_span to keep the JSON short.\n"
                "- If no relation is supported, return an empty triples array.\n"
            )
        return (
            "Output compact valid JSON with this exact top-level shape:\n"
            "{\n"
            '  "passages": [\n'
            '    {"passage_id": "string", '
            '"triples": [{"head": "string", "relation": "string", "tail": "string", "factual_confidence": 0.0, "semantic_salience": 0.0, "bridge_potential": 0.0}], '
            '"high_order_relations": [{"relation": "string", "entities": ["string"], "factual_confidence": 0.0, "semantic_salience": 0.0, "bridge_potential": 0.0}]}\n'
            "  ],\n"
            '  "cross_passage_high_order_relations": []\n'
            "}\n\n"
            "Rules:\n"
            "- Return JSON only; no markdown and no comments.\n"
            "- Use only passage_id values provided below.\n"
            "- Prefer compact canonical entity names.\n"
            f"- Extract at most {self.max_triples_per_passage} salient triples per passage.\n"
            f"- Extract at most {self.max_high_order_per_passage} high_order_relations per passage.\n"
            "- A high_order_relation must contain at least 3 entities.\n"
            "- factual_confidence: 0 means weak textual support; 1 means explicitly stated in the passage.\n"
            "- semantic_salience: 0 means peripheral detail; 1 means central relation/event of the passage.\n"
            "- bridge_potential: 0 means unlikely to help multi-hop linking; 1 means connects entities/events/attributes that could bridge to other passages.\n"
            "- Do not set all scores to the same value; use the full 0-1 range when justified.\n"
            "- Omit source_span to keep the JSON short.\n"
            "- If no relation is supported, return empty arrays.\n"
        )

    def _example_user_prompt(self, ex: QAExample) -> str:
        passages = []
        for passage in ex.passages:
            text = passage.full_text()[: self.max_passage_chars]
            passages.append({"passage_id": passage.id, "title": passage.title, "text": text})
        question_block = f'\nQuestion for optional query-aware relevance: "{ex.question}"\n' if self.include_question else "\n"
        return (
            self._schema_prompt()
            + f"{question_block}"
            + f"Example id: {ex.id}\n"
            + f"Passages json:\n{json.dumps(passages, ensure_ascii=False)}"
        )

    def _passage_user_prompt(self, ex: QAExample, passage: Passage) -> str:
        question_block = f'\nQuestion for optional query-aware relevance: "{ex.question}"\n' if self.include_question else "\n"
        passage_obj = {
            "passage_id": passage.id,
            "title": passage.title,
            "text": passage.full_text()[: self.max_passage_chars],
        }
        return (
            self._schema_prompt()
            + f"{question_block}"
            + f"Example id: {ex.id}\n"
            + f"Passage json:\n{json.dumps(passage_obj, ensure_ascii=False)}"
        )

    def _parse_json(self, text: str) -> dict[str, Any]:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise
            obj = json.loads(text[start : end + 1])
        if not isinstance(obj, dict):
            raise ValueError("DeepSeek OpenIE response must be a JSON object.")
        return obj

    def _to_artifacts(self, ex: QAExample, parsed: dict[str, Any]) -> tuple[list[Triple], list[Hyperedge]]:
        valid_passage_ids = {p.id for p in ex.passages}
        triples: list[Triple] = []
        hyperedges: list[Hyperedge] = []
        for passage_obj in parsed.get("passages", []) or []:
            if not isinstance(passage_obj, dict):
                continue
            pid = str(passage_obj.get("passage_id", ""))
            if pid not in valid_passage_ids:
                continue
            for item in passage_obj.get("triples", []) or []:
                triple = self._triple_from_obj(item, pid)
                if triple:
                    triples.append(triple)
            if self.extract_high_order_relations:
                for item in passage_obj.get("high_order_relations", []) or []:
                    edge = self._hyperedge_from_obj(ex.id, item, [pid], "llm_high_order")
                    if edge:
                        hyperedges.append(edge)
        if self.extract_high_order_relations:
            for item in parsed.get("cross_passage_high_order_relations", []) or []:
                if not isinstance(item, dict):
                    continue
                pids = [str(pid) for pid in item.get("passage_ids", []) if str(pid) in valid_passage_ids]
                edge = self._hyperedge_from_obj(ex.id, item, pids, "llm_cross_passage")
                if edge:
                    hyperedges.append(edge)
        return self._dedupe_triples(triples), self._dedupe_hyperedges(hyperedges)

    def _triple_from_obj(self, item: Any, passage_id: str) -> Triple | None:
        if not isinstance(item, dict):
            return None
        head = _canonical_text(item.get("head"))
        tail = _canonical_text(item.get("tail"))
        relation = _canonical_text(item.get("relation")) or "related_to"
        if not head or not tail or head.lower() == tail.lower():
            return None
        factual = _field_score(item, "factual_confidence", default=0.75)
        salience = _field_score(item, "semantic_salience", default=0.5)
        bridge = _field_score(item, "bridge_potential", default=0.25)
        return Triple(
            head=head,
            relation=relation,
            tail=tail,
            passage_id=passage_id,
            confidence=factual,
            source=self.model,
            factual_confidence=factual,
            semantic_salience=salience,
            bridge_potential=bridge,
        )

    def _hyperedge_from_obj(self, example_id: str, item: Any, passage_ids: list[str], source: str) -> Hyperedge | None:
        if not isinstance(item, dict):
            return None
        vertices = tuple(sorted({_canonical_text(v) for v in item.get("entities", []) if _canonical_text(v)}))
        pids = tuple(sorted(set(passage_ids)))
        if len(vertices) < 3 and len(pids) < 2:
            return None
        relation = _canonical_text(item.get("relation")) or source
        factual = _field_score(item, "factual_confidence", default=0.75)
        salience = _field_score(item, "semantic_salience", default=0.6)
        bridge = _field_score(item, "bridge_potential", default=0.5)
        confidence = factual
        return Hyperedge(
            id=_stable_id("llm_he", example_id, relation, vertices, pids, source),
            vertices=vertices,
            passage_ids=pids,
            relation=relation,
            bridge=relation.lower(),
            weight=1.0,
            confidence=confidence,
            source=source,
            factual_confidence=factual,
            semantic_salience=salience,
            bridge_potential=bridge,
        )

    def _dedupe_triples(self, triples: list[Triple]) -> list[Triple]:
        seen: set[tuple[str, str, str, str]] = set()
        out: list[Triple] = []
        for triple in triples:
            key = (triple.head.lower(), triple.relation.lower(), triple.tail.lower(), triple.passage_id)
            if key in seen:
                continue
            seen.add(key)
            out.append(triple)
        return out

    def _dedupe_hyperedges(self, edges: list[Hyperedge]) -> list[Hyperedge]:
        seen: set[tuple[tuple[str, ...], tuple[str, ...], str]] = set()
        out: list[Hyperedge] = []
        for edge in edges:
            key = (tuple(v.lower() for v in edge.vertices), edge.passage_ids, edge.relation.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(edge)
        return out
