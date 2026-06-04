from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any

from .schema import QAExample, RetrievalOutput
from .utils import normalize_text, tokenize


def exact_match(prediction: str, answers: list[str]) -> float:
    pred = normalize_text(prediction)
    return float(any(pred == normalize_text(a) for a in answers))


def f1_score(prediction: str, answers: list[str]) -> float:
    pred_toks = tokenize(normalize_text(prediction))
    best = 0.0
    for answer in answers:
        gold_toks = tokenize(normalize_text(answer))
        if not pred_toks and not gold_toks:
            best = max(best, 1.0)
            continue
        if not pred_toks or not gold_toks:
            continue
        common = 0
        used = [False] * len(gold_toks)
        for tok in pred_toks:
            for i, gt in enumerate(gold_toks):
                if not used[i] and tok == gt:
                    used[i] = True
                    common += 1
                    break
        if common == 0:
            continue
        precision = common / len(pred_toks)
        recall = common / len(gold_toks)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def retrieval_metrics(ex: QAExample, ranked_ids: list[str]) -> dict[str, float]:
    gold = set(ex.gold_passage_ids)
    out: dict[str, float] = {}
    if not gold:
        for k in (5, 10):
            out[f"recall@{k}"] = 0.0
            out[f"all_recall@{k}"] = 0.0
        out["mrr"] = 0.0
        out["gold_rank"] = 0.0
        return out
    for k in (5, 10):
        top = set(ranked_ids[:k])
        out[f"recall@{k}"] = 100.0 * len(top & gold) / len(gold)
        out[f"all_recall@{k}"] = 100.0 * float(gold.issubset(top))
    ranks = [rank + 1 for rank, pid in enumerate(ranked_ids) if pid in gold]
    out["mrr"] = 1.0 / min(ranks) if ranks else 0.0
    out["gold_rank"] = float(min(ranks)) if ranks else 0.0
    return out


def answer_support_rate(ex: QAExample, context: str) -> float:
    ctx = normalize_text(context)
    return float(any(normalize_text(a) in ctx for a in ex.answers if a))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    numeric: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                numeric[key].append(float(value))
    out: dict[str, float] = {}
    for key, values in numeric.items():
        out[key] = mean(values) if values else 0.0
    return out


def mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": mean(values), "std": pstdev(values) if len(values) > 1 else 0.0}


def go_no_go(summary: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    rule = cfg["go_no_go"]
    ref = rule["reference_method"]
    cand = rule["candidate_method"]
    datasets = sorted(summary.get("datasets", {}).keys())
    checks: list[dict[str, Any]] = []
    passed_all = True
    for dataset in datasets:
        methods = summary["datasets"][dataset]["methods"]
        if ref not in methods or cand not in methods:
            checks.append({"dataset": dataset, "status": "missing_method"})
            passed_all = False
            continue
        r = methods[ref]["mean"]
        c = methods[cand]["mean"]
        all_recall_gain = c.get("all_recall@10", 0.0) - r.get("all_recall@10", 0.0)
        f1_gain = c.get("f1", 0.0) - r.get("f1", 0.0)
        em_gain = c.get("em", 0.0) - r.get("em", 0.0)
        latency_ratio = c.get("latency_seconds", 0.0) / max(1e-9, r.get("latency_seconds", 0.0))
        pass_quality = all_recall_gain >= float(rule["all_recall_at_10_gain"]) and (
            f1_gain >= float(rule["qa_f1_gain"]) or em_gain >= float(rule["em_gain"])
        )
        pass_latency = latency_ratio <= float(rule["latency_multiplier"])
        status = "go" if pass_quality and pass_latency else "optimize_latency" if pass_quality else "no_go"
        if status != "go":
            passed_all = False
        checks.append(
            {
                "dataset": dataset,
                "all_recall@10_gain": all_recall_gain,
                "f1_gain": f1_gain,
                "em_gain": em_gain,
                "latency_ratio": latency_ratio,
                "status": status,
            }
        )
    return {"passed_all": passed_all, "checks": checks}

