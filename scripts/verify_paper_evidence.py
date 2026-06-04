#!/usr/bin/env python3
"""Verify paper-facing HKVM-RAG evidence tables against staged artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PAPER_EVIDENCE = ROOT / "results" / "paper_evidence"
FROZEN_INDEX = ROOT / "frozen_outputs" / "manifests" / "frozen_output_index.json"


def read_rows(rel: str) -> list[dict[str, str]]:
    with (PAPER_EVIDENCE / rel).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def r3(value: str | float) -> float:
    return round(float(value), 3)


def add_check(checks: list[tuple[str, str, Any, Any, bool]], group: str, key: str, got: Any, expected: Any) -> None:
    checks.append((group, key, got, expected, got == expected))


def verify() -> list[tuple[str, str, Any, Any, bool]]:
    checks: list[tuple[str, str, Any, Any, bool]] = []

    main = read_rows("main_fixed_substrate/table_main.csv")
    for dataset, expected in [("2wiki", 3.426), ("musique", 3.592), ("hotpotqa", -0.689)]:
        by_method = {row["method"]: row for row in main if row["dataset"] == dataset}
        got = r3(float(by_method["weighted_hg_kv"]["f1"]) - float(by_method["kg_ppr"]["f1"]))
        add_check(checks, "fixed_substrate_delta_f1", dataset, got, r3(expected))

    controller = read_rows("adaptive_source_controller/table.csv")
    for dataset, variant, metric, expected in [
        ("musique", "colbertv2", "f1", 58.309),
        ("musique", "controller_dense_hkvm", "f1", 65.073),
        ("2wiki", "colbertv2", "f1", 77.763),
        ("2wiki", "controller_dense_hkvm", "f1", 88.846),
        ("hotpotqa", "colbertv2", "f1", 79.844),
        ("hotpotqa", "controller_dense_hkvm", "f1", 85.810),
        ("musique", "controller_dense_hkvm", "em", 64.722),
        ("2wiki", "controller_dense_hkvm", "em", 88.658),
        ("hotpotqa", "controller_dense_hkvm", "em", 85.627),
    ]:
        row = next(
            item
            for item in controller
            if item["dataset"] == dataset and item["variant"] == variant and item["metric"] == metric
        )
        add_check(checks, "adaptive_controller_table", f"{dataset}/{variant}/{metric}", r3(row["mean"]), r3(expected))

    boot = read_rows("adaptive_source_controller/paired_bootstrap/summary.csv")
    for dataset, reference, expected_diff, expected_low, expected_high in [
        ("musique", "colbertv2", 6.763, 5.791, 7.726),
        ("musique", "calibrated_hkvm_topk4", 8.319, 7.327, 9.293),
        ("2wiki", "colbertv2", 11.084, 10.740, 11.437),
        ("2wiki", "calibrated_hkvm_topk4", 0.455, 0.368, 0.542),
        ("hotpotqa", "colbertv2", 5.966, 5.491, 6.452),
        ("hotpotqa", "calibrated_hkvm_topk4", 2.881, 2.555, 3.201),
    ]:
        row = next(
            item
            for item in boot
            if item["dataset"] == dataset
            and item["reference_variant"] == reference
            and item["candidate_variant"] == "controller_dense_hkvm"
            and item["metric"] == "f1"
            and item["bootstrap_unit"] == "seed_example"
        )
        got = (r3(row["mean_diff"]), r3(row["ci_low"]), r3(row["ci_high"]))
        expected = (r3(expected_diff), r3(expected_low), r3(expected_high))
        add_check(checks, "adaptive_controller_bootstrap_f1", f"{dataset}/{reference}", got, expected)

    source_expected = {
        "colbertv2": {
            "musique": (58.309, 56.853, 65.073),
            "2wiki": (77.763, 80.769, 88.846),
            "hotpotqa": (79.844, 80.413, 85.810),
        },
        "bm25": {
            "musique": (46.045, 45.872, 58.083),
            "2wiki": (71.873, 78.539, 88.325),
            "hotpotqa": (81.835, 82.618, 86.806),
        },
        "contriever": {
            "musique": (55.565, 55.773, 63.378),
            "2wiki": (65.498, 79.773, 88.618),
            "hotpotqa": (79.930, 79.574, 85.096),
        },
    }
    for source, expected_by_dataset in source_expected.items():
        table = read_rows(f"source_level_ablation/{source}/table.csv")
        for dataset, expected in expected_by_dataset.items():
            rows = [row for row in table if row["dataset"] == dataset and row["metric"] == "f1"]
            base = next(row for row in rows if row["row_type"] == "dense_baseline")
            whg = next(row for row in rows if row["source"] == "weighted_hg_kv")
            non_whg = max(
                [row for row in rows if row["source"] in {"hyper_rag", "static_hg", "weighted_kg_ppr"}],
                key=lambda row: float(row["mean"]),
            )
            got = (r3(base["mean"]), r3(non_whg["mean"]), r3(whg["mean"]))
            add_check(checks, "source_robustness_f1", f"{source}/{dataset}", got, tuple(r3(x) for x in expected))

    source_bootstrap_expected = {
        ("colbertv2", "musique"): (6.763, 5.813, 7.753),
        ("bm25", "musique"): (12.038, 10.935, 13.120),
        ("contriever", "musique"): (7.813, 6.810, 8.814),
        ("colbertv2", "2wiki"): (11.084, 10.722, 11.442),
        ("bm25", "2wiki"): (16.452, 16.039, 16.865),
        ("contriever", "2wiki"): (23.119, 22.668, 23.577),
        ("colbertv2", "hotpotqa"): (5.966, 5.479, 6.456),
        ("bm25", "hotpotqa"): (4.971, 4.554, 5.374),
        ("contriever", "hotpotqa"): (5.166, 4.683, 5.634),
    }
    for (source, dataset), expected in source_bootstrap_expected.items():
        table = read_rows(f"source_level_ablation/{source}/bootstrap.csv")
        row = next(
            item
            for item in table
            if item["dataset"] == dataset
            and item["reference_variant"] == source
            and item["candidate_source"] == "weighted_hg_kv"
            and item["candidate_variant"] == "controller_dense_hkvm"
            and item["metric"] == "f1"
        )
        got = (r3(row["pooled_mean_diff"]), r3(row["pooled_ci_low"]), r3(row["pooled_ci_high"]))
        add_check(checks, "source_robustness_bootstrap_f1", f"{source}/{dataset}", got, tuple(r3(x) for x in expected))

    frozen = json.loads(FROZEN_INDEX.read_text(encoding="utf-8"))["entries"]
    coverage = {(item["experiment_family"], item["dataset"], item["method"]) for item in frozen}
    required = []
    for dataset in ("2wiki", "musique", "hotpotqa"):
        for method in ("bm25", "kg_ppr", "weighted_hg_kv"):
            required.append(("fixed_substrate", dataset, method))
    for dataset in ("2wiki", "musique", "hotpotqa"):
        for method in (
            "colbertv2",
            "calibrated_hkvm_topk4",
            "controller_dense_hkvm",
            "controller_dense_only",
            "controller_hkvm_only",
            "rrf_dense_hkvm",
        ):
            required.append(("adaptive_source_controller", dataset, method))
    for item in required:
        add_check(checks, "frozen_output_coverage", "/".join(item), "present" if item in coverage else "missing", "present")

    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify HKVM-RAG paper evidence files.")
    parser.add_argument("--json", action="store_true", help="Print full check records as JSON.")
    args = parser.parse_args()
    checks = verify()
    bad = [item for item in checks if not item[-1]]
    if args.json:
        print(json.dumps({"checks": checks, "bad": bad}, indent=2, ensure_ascii=False))
    else:
        grouped: dict[str, int] = defaultdict(int)
        for group, *_ in checks:
            grouped[group] += 1
        print(f"checks={len(checks)} bad={len(bad)}")
        print("groups=" + json.dumps(dict(sorted(grouped.items())), sort_keys=True))
        for item in bad:
            print("BAD", item)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
