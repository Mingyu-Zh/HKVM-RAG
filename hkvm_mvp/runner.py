from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .config import load_config
from .data import load_examples
from .extraction import build_artifacts
from .generation import ExtractiveGenerator
from .methods import METHODS, MethodContext
from .metrics import aggregate, answer_support_rate, exact_match, f1_score, go_no_go, mean_std, retrieval_metrics
from .reproducibility import (
    build_run_manifest,
    extraction_audit_summary,
    finalize_run_manifest,
    paired_bootstrap_report,
    write_efficiency_report,
)
from .schema import ExtractionArtifacts, QAExample
from .utils import ensure_dir, set_seed, token_count, write_json, write_jsonl


STRUCTURED_METHODS = {"kg_ppr", "weighted_kg_ppr", "hyper_rag", "static_hg", "weighted_hg_kv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HKVM-RAG MVP experiments.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset keys.")
    parser.add_argument("--methods", default=None, help="Comma-separated method names.")
    parser.add_argument("--runs", type=int, default=None, help="Override number of seeds.")
    parser.add_argument("--limit", type=int, default=None, help="Override per-dataset example limit.")
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def _selected(config: dict[str, Any], args: argparse.Namespace) -> tuple[list[str], list[str], list[int]]:
    datasets = args.datasets.split(",") if args.datasets else list(config["datasets"].keys())
    methods = args.methods.split(",") if args.methods else list(config["methods"])
    seeds = list(config["runs"]["seeds"])
    if args.runs:
        seeds = seeds[: args.runs]
    return datasets, methods, seeds


def _run_method(examples: list[QAExample], method_name: str, ctx: MethodContext, out_dir: Path) -> dict[str, float]:
    method_cls = METHODS[method_name]
    method = method_cls(ctx)
    generator = ExtractiveGenerator(int(ctx.config["matched_budget"]["max_context_tokens"]))
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for ex in tqdm(examples, desc=f"{method_name}", leave=False):
        ret = method.retrieve(ex)
        pred, context = generator.generate(ex, ret.ranked_passage_ids[: int(ctx.config["matched_budget"]["generation_top_k"])])
        row = retrieval_metrics(ex, ret.ranked_passage_ids)
        row["em"] = 100.0 * exact_match(pred, ex.answers)
        row["f1"] = 100.0 * f1_score(pred, ex.answers)
        row["answer_support_rate"] = 100.0 * answer_support_rate(ex, context)
        row["latency_seconds"] = ret.latency_seconds
        row["avg_retrieval_tokens"] = token_count(context)
        rows.append(row)
        predictions.append(
            {
                "id": ex.id,
                "question": ex.question,
                "answers": ex.answers,
                "prediction": pred,
                "ranked_passage_ids": ret.ranked_passage_ids,
                "scores": ret.scores,
                "metrics": row,
                "debug": ret.debug,
            }
        )
    metrics = aggregate(rows)
    write_jsonl(out_dir / f"{method_name}.predictions.jsonl", predictions)
    write_json(out_dir / f"{method_name}.metrics.json", metrics)
    return metrics


def _summarize(all_metrics: dict[str, dict[str, dict[int, dict[str, float]]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"datasets": {}}
    for dataset, method_data in all_metrics.items():
        summary["datasets"][dataset] = {"methods": {}}
        for method, seed_data in method_data.items():
            keys = sorted({k for metrics in seed_data.values() for k in metrics})
            mean_obj: dict[str, float] = {}
            std_obj: dict[str, float] = {}
            for key in keys:
                stats = mean_std([metrics.get(key, 0.0) for metrics in seed_data.values()])
                mean_obj[key] = stats["mean"]
                std_obj[key] = stats["std"]
            summary["datasets"][dataset]["methods"][method] = {"mean": mean_obj, "std": std_obj, "seeds": seed_data}
    return summary


def _write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "method", "metric", "mean", "std"])
        for dataset, d_obj in summary["datasets"].items():
            for method, m_obj in d_obj["methods"].items():
                for metric, value in sorted(m_obj["mean"].items()):
                    writer.writerow([dataset, method, metric, value, m_obj["std"].get(metric, 0.0)])


def _write_go_markdown(go: dict[str, Any], path: Path) -> None:
    lines = ["# Go/No-Go Summary", "", "| Dataset | All-Recall@10 Gain | F1 Gain | EM Gain | Latency Ratio | Status |", "|---|---:|---:|---:|---:|---|"]
    for check in go["checks"]:
        lines.append(
            f"| {check.get('dataset')} | {check.get('all_recall@10_gain', 0):.3f} | "
            f"{check.get('f1_gain', 0):.3f} | {check.get('em_gain', 0):.3f} | "
            f"{check.get('latency_ratio', 0):.3f} | {check.get('status')} |"
        )
    lines.append("")
    lines.append(f"Overall passed: `{go['passed_all']}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def _empty_artifacts() -> ExtractionArtifacts:
    return ExtractionArtifacts(
        triples_by_example={},
        hyperedges_by_example={},
        cooccurrence_hyperedges_by_example={},
        random_hyperedges_by_example={},
        gold_support_by_example={},
    )


def _skipped_extraction_audit(examples: list[QAExample], methods: list[str]) -> dict[str, Any]:
    return {
        "skipped": True,
        "reason": "No selected method requires OpenIE / graph / hypergraph artifacts.",
        "methods": methods,
        "examples": len(examples),
        "passages": sum(len(ex.passages) for ex in examples),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["project"]["output_dir"] = args.output_dir
    datasets, methods, seeds = _selected(config, args)
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method {method}. Valid: {sorted(METHODS)}")
    out_root = ensure_dir(config["project"]["output_dir"])
    dataset_cfgs: dict[str, dict[str, Any]] = {}
    for dataset_name in datasets:
        dataset_cfgs[dataset_name] = dict(config["datasets"][dataset_name])
        if args.limit:
            dataset_cfgs[dataset_name]["limit"] = args.limit
    manifest_path = out_root / "run_manifest.json"
    manifest = build_run_manifest(config, args, datasets, methods, seeds, dataset_cfgs, out_root)
    write_json(manifest_path, manifest)
    all_metrics: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    extraction_audits: dict[str, dict[str, Any]] = {}
    try:
        for dataset_name in datasets:
            dataset_cfg = dataset_cfgs[dataset_name]
            examples = load_examples(dataset_name, dataset_cfg)
            all_metrics[dataset_name] = {m: {} for m in methods}
            extraction_audits[dataset_name] = {}
            for seed in seeds:
                print(f"[hkvm-rag] Running {dataset_name} seed={seed}.")
                set_seed(seed)
                if any(method in STRUCTURED_METHODS for method in methods):
                    artifacts = build_artifacts(examples, dataset_name, config, seed, dataset_cfg=dataset_cfg)
                    audit = extraction_audit_summary(examples, artifacts)
                else:
                    artifacts = _empty_artifacts()
                    audit = _skipped_extraction_audit(examples, methods)
                extraction_audits[dataset_name][str(seed)] = audit
                ctx = MethodContext(config=config, artifacts=artifacts, seed=seed)
                run_dir = ensure_dir(out_root / "runs" / dataset_name / f"seed_{seed}")
                write_json(run_dir / "extraction_audit.json", audit)
                for method_name in methods:
                    metrics = _run_method(examples, method_name, ctx, run_dir)
                    all_metrics[dataset_name][method_name][seed] = metrics
                    print(f"[hkvm-rag] {dataset_name}/{method_name}/seed={seed}: {metrics}")
        summary = _summarize(all_metrics)
        write_json(out_root / "summary.json", summary)
        _write_summary_csv(summary, out_root / "summary.csv")
        write_json(out_root / "extraction_audit.json", extraction_audits)
        go = go_no_go(summary, config)
        write_json(out_root / "go_no_go.json", go)
        _write_go_markdown(go, out_root / "go_no_go.md")
        efficiency = write_efficiency_report(summary, out_root / "efficiency.json")
        bootstrap = paired_bootstrap_report(out_root, datasets, seeds, config)
        write_json(out_root / "bootstrap.json", bootstrap)
        finalize_run_manifest(
            manifest_path,
            "completed",
            {
                "outputs": {
                    "summary": str(out_root / "summary.json"),
                    "summary_csv": str(out_root / "summary.csv"),
                    "go_no_go": str(out_root / "go_no_go.json"),
                    "bootstrap": str(out_root / "bootstrap.json"),
                    "efficiency": str(out_root / "efficiency.json"),
                    "extraction_audit": str(out_root / "extraction_audit.json"),
                },
                "go_no_go": go,
                "efficiency": efficiency,
                "bootstrap": bootstrap,
            },
        )
        print(f"[hkvm-rag] Wrote summary to {out_root / 'summary.json'}")
        print(f"[hkvm-rag] Go/No-Go: {go}")
    except Exception as exc:
        finalize_run_manifest(manifest_path, "failed", {"error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
