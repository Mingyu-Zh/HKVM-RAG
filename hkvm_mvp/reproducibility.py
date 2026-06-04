from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .schema import ExtractionArtifacts, QAExample
from .utils import ensure_dir, read_json, read_jsonl, write_json


def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(obj: Any, length: int = 16) -> str:
    return hashlib.sha256(stable_json(obj).encode("utf-8")).hexdigest()[:length]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _without_keys(obj: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {k: v for k, v in obj.items() if k not in keys}


def dataset_fingerprint(dataset_name: str, dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    path = dataset_cfg.get("path")
    payload: dict[str, Any] = {
        "dataset": dataset_name,
        "loader": dataset_cfg.get("loader"),
        "path": path,
        "hf_name": dataset_cfg.get("hf_name"),
        "hf_config": dataset_cfg.get("hf_config"),
        "split": dataset_cfg.get("split"),
        "answerable_only": dataset_cfg.get("answerable_only", False),
        "limit": dataset_cfg.get("limit"),
    }
    if path and Path(path).exists():
        p = Path(path)
        payload["file_size_bytes"] = p.stat().st_size
        payload["file_sha256"] = file_sha256(p)
    return payload


def hyperedge_mode(config: dict[str, Any]) -> dict[str, Any]:
    ab = config.get("ablations", {})
    keys = [
        "cooccurrence_hg",
        "answer_path_hg",
        "low_order_only",
        "high_order_only",
        "all_hyperedges",
        "gold_hyperedges",
        "llm_extracted_hyperedges",
        "random_hyperedges",
        "diffusion_steps_1",
        "diffusion_steps_2",
    ]
    return {k: ab.get(k) for k in keys}


def extractor_config(config: dict[str, Any]) -> dict[str, Any]:
    openie = _without_keys(dict(config.get("openie", {})), {"cache_dir", "api_key"})
    matched = config.get("matched_budget", {})
    retrieval = config.get("retrieval", {})
    return {
        "openie": openie,
        "chunking": {
            "chunk_size_tokens": matched.get("chunk_size_tokens"),
            "chunk_overlap_tokens": matched.get("chunk_overlap_tokens"),
        },
        "random_hyperedges": {
            "random_hyperedge_count": retrieval.get("random_hyperedge_count"),
            "random_hyperedge_size": retrieval.get("random_hyperedge_size"),
        },
        "hyperedge_mode": hyperedge_mode(config),
    }


def cache_context(config: dict[str, Any], dataset_name: str, dataset_cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    openie = config.get("openie", {})
    cache_seed: int | str = "seed_invariant" if bool(openie.get("seed_invariant_cache", False)) else seed
    key = {
        "dataset": dataset_fingerprint(dataset_name, dataset_cfg),
        "extractor_id": openie.get("backend", "heuristic"),
        "extractor_config_hash": stable_hash(extractor_config(config), length=24),
        "hyperedge_mode": hyperedge_mode(config),
        "seed": cache_seed,
    }
    return {
        "cache_hash": stable_hash(key, length=24),
        "cache_key": key,
    }


def validate_cache_manifest(cache_path: str | Path, expected_context: dict[str, Any]) -> None:
    path = Path(cache_path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise RuntimeError(
            f"Cache file exists without manifest: {path}. "
            "Delete it or rebuild with cache fingerprinting enabled."
        )
    manifest = read_json(manifest_path)
    expected_hash = expected_context.get("cache_hash")
    if manifest.get("cache_hash") != expected_hash or manifest.get("cache_key") != expected_context.get("cache_key"):
        raise RuntimeError(
            "OpenIE cache manifest mismatch. "
            f"cache={path}, expected_hash={expected_hash}, found_hash={manifest.get('cache_hash')}"
        )


def write_cache_manifest(cache_path: str | Path, context: dict[str, Any], stats: dict[str, Any]) -> None:
    path = Path(cache_path)
    manifest = {
        "cache_file": str(path),
        "cache_hash": context.get("cache_hash"),
        "cache_key": context.get("cache_key"),
        "stats": stats,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(path.with_suffix(path.suffix + ".manifest.json"), manifest)


def code_fingerprint(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    files: list[Path] = []
    for pattern in ("*.py", "*.yaml", "hkvm_mvp/*.py"):
        files.extend(root.glob(pattern))
    file_hashes: dict[str, str] = {}
    for path in sorted(set(files)):
        if path.is_file():
            file_hashes[str(path.relative_to(root))] = file_sha256(path)
    return {"root": str(root), "files": file_hashes, "hash": stable_hash(file_hashes, length=24)}


def build_run_manifest(
    config: dict[str, Any],
    args: argparse.Namespace,
    datasets: list[str],
    methods: list[str],
    seeds: list[int],
    dataset_cfgs: dict[str, dict[str, Any]],
    out_root: str | Path,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    return {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ended_at": None,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "output_dir": str(out_root),
        "config_hash": stable_hash(config, length=24),
        "extractor_config_hash": stable_hash(extractor_config(config), length=24),
        "hyperedge_mode": hyperedge_mode(config),
        "code": code_fingerprint(project_root),
        "selection": {
            "datasets": datasets,
            "methods": methods,
            "seeds": seeds,
            "args": vars(args),
        },
        "datasets": {name: dataset_fingerprint(name, cfg) for name, cfg in dataset_cfgs.items()},
    }


def finalize_run_manifest(path: str | Path, status: str, extra: dict[str, Any] | None = None) -> None:
    manifest_path = Path(path)
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    manifest["status"] = status
    manifest["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if extra:
        manifest.update(extra)
    write_json(manifest_path, manifest)


def extraction_audit_summary(examples: list[QAExample], artifacts: ExtractionArtifacts) -> dict[str, Any]:
    passage_count = sum(len(ex.passages) for ex in examples)
    triple_count = sum(len(v) for v in artifacts.triples_by_example.values())
    answer_path_count = sum(len(v) for v in artifacts.hyperedges_by_example.values())
    co_count = sum(len(v) for v in artifacts.cooccurrence_hyperedges_by_example.values())
    random_count = sum(len(v) for v in artifacts.random_hyperedges_by_example.values())
    empty_examples = sum(1 for ex in examples if not artifacts.triples_by_example.get(ex.id))
    gold_examples = sum(1 for ex in examples if ex.gold_passage_ids)
    triples = [t for vals in artifacts.triples_by_example.values() for t in vals]
    hyperedges = [h for vals in artifacts.hyperedges_by_example.values() for h in vals]

    def dist(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0}
        arr = np.array(values, dtype=float)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "p25": float(np.quantile(arr, 0.25)),
            "p50": float(np.quantile(arr, 0.50)),
            "p75": float(np.quantile(arr, 0.75)),
            "max": float(arr.max()),
            "unique_rounded": sorted({round(float(x), 2) for x in values})[:25],
        }

    return {
        "examples": len(examples),
        "passages": passage_count,
        "triples": triple_count,
        "answer_path_hyperedges": answer_path_count,
        "cooccurrence_hyperedges": co_count,
        "random_hyperedges": random_count,
        "triples_per_passage": triple_count / max(1, passage_count),
        "triples_per_example": triple_count / max(1, len(examples)),
        "answer_path_hyperedges_per_example": answer_path_count / max(1, len(examples)),
        "cooccurrence_hyperedges_per_example": co_count / max(1, len(examples)),
        "empty_extraction_examples": empty_examples,
        "empty_extraction_rate": empty_examples / max(1, len(examples)),
        "gold_support_examples": gold_examples,
        "gold_support_coverage": gold_examples / max(1, len(examples)),
        "triple_score_distributions": {
            "factual_confidence": dist([t.factual_confidence for t in triples]),
            "semantic_salience": dist([t.semantic_salience for t in triples]),
            "bridge_potential": dist([t.bridge_potential for t in triples]),
        },
        "hyperedge_score_distributions": {
            "factual_confidence": dist([h.factual_confidence for h in hyperedges]),
            "semantic_salience": dist([h.semantic_salience for h in hyperedges]),
            "bridge_potential": dist([h.bridge_potential for h in hyperedges]),
        },
    }


def write_efficiency_report(summary: dict[str, Any], path: str | Path) -> dict[str, Any]:
    report: dict[str, Any] = {"datasets": {}}
    for dataset, d_obj in summary.get("datasets", {}).items():
        report["datasets"][dataset] = {}
        for method, m_obj in d_obj.get("methods", {}).items():
            mean = m_obj.get("mean", {})
            std = m_obj.get("std", {})
            report["datasets"][dataset][method] = {
                "latency_seconds_mean": mean.get("latency_seconds", 0.0),
                "latency_seconds_std": std.get("latency_seconds", 0.0),
                "avg_retrieval_tokens_mean": mean.get("avg_retrieval_tokens", 0.0),
                "avg_retrieval_tokens_std": std.get("avg_retrieval_tokens", 0.0),
            }
    write_json(path, report)
    return report


def _prediction_metric_rows(path: Path) -> dict[str, dict[str, float]]:
    rows = {}
    for row in read_jsonl(path):
        metrics = row.get("metrics", {})
        rows[str(row.get("id"))] = {k: float(metrics.get(k, 0.0)) for k in ("all_recall@10", "f1", "em")}
    return rows


def paired_bootstrap_report(out_root: str | Path, datasets: list[str], seeds: list[int], cfg: dict[str, Any]) -> dict[str, Any]:
    eval_cfg = cfg.get("evaluation", {})
    samples = int(eval_cfg.get("bootstrap_samples", 1000))
    alpha = float(eval_cfg.get("bootstrap_alpha", 0.05))
    if samples <= 0:
        return {"enabled": False, "reason": "evaluation.bootstrap_samples <= 0"}
    ref = cfg["go_no_go"]["reference_method"]
    cand = cfg["go_no_go"]["candidate_method"]
    rng = np.random.default_rng(int(eval_cfg.get("bootstrap_seed", 13)))
    report: dict[str, Any] = {"enabled": True, "samples": samples, "alpha": alpha, "comparisons": []}
    for dataset in datasets:
        diffs_by_metric: dict[str, list[float]] = {"all_recall@10": [], "f1": [], "em": []}
        missing: list[str] = []
        for seed in seeds:
            run_dir = Path(out_root) / "runs" / dataset / f"seed_{seed}"
            ref_path = run_dir / f"{ref}.predictions.jsonl"
            cand_path = run_dir / f"{cand}.predictions.jsonl"
            if not ref_path.exists() or not cand_path.exists():
                missing.append(f"{dataset}/seed_{seed}")
                continue
            ref_rows = _prediction_metric_rows(ref_path)
            cand_rows = _prediction_metric_rows(cand_path)
            for ex_id in sorted(set(ref_rows) & set(cand_rows)):
                for metric in diffs_by_metric:
                    diffs_by_metric[metric].append(cand_rows[ex_id][metric] - ref_rows[ex_id][metric])
        comp: dict[str, Any] = {"dataset": dataset, "reference": ref, "candidate": cand, "missing": missing, "metrics": {}}
        for metric, values in diffs_by_metric.items():
            arr = np.array(values, dtype=float)
            if arr.size == 0:
                comp["metrics"][metric] = {"n": 0}
                continue
            boot = np.empty(samples, dtype=float)
            for i in range(samples):
                idx = rng.integers(0, arr.size, size=arr.size)
                boot[i] = float(arr[idx].mean())
            comp["metrics"][metric] = {
                "n": int(arr.size),
                "mean_diff": float(arr.mean()),
                "ci_low": float(np.quantile(boot, alpha / 2.0)),
                "ci_high": float(np.quantile(boot, 1.0 - alpha / 2.0)),
                "p_diff_le_0": float(np.mean(boot <= 0.0)),
            }
        report["comparisons"].append(comp)
    return report
