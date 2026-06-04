from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .utils import ensure_dir, read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot HKVM-RAG MVP results.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def _metric_bars(summary: dict, metric: str, path: Path) -> None:
    labels: list[str] = []
    values: list[float] = []
    errors: list[float] = []
    for dataset, d_obj in summary["datasets"].items():
        for method, m_obj in d_obj["methods"].items():
            labels.append(f"{dataset}\n{method}")
            values.append(m_obj["mean"].get(metric, 0.0))
            errors.append(m_obj["std"].get(metric, 0.0))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    ax.bar(range(len(labels)), values, yerr=errors, capsize=3)
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(f"HKVM-RAG MVP {metric} (mean ± std)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary = read_json(args.summary)
    out_dir = ensure_dir(args.out_dir)
    _metric_bars(summary, "all_recall@10", out_dir / "Figure_1.png")
    _metric_bars(summary, "f1", out_dir / "Figure_2.png")


if __name__ == "__main__":
    main()

