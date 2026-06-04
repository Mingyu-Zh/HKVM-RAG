from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import Passage, QAExample
from .utils import read_jsonl


def _as_answers(record: dict[str, Any]) -> list[str]:
    answers = record.get("answers", record.get("answer", []))
    aliases = record.get("answer_aliases", [])
    if isinstance(answers, str):
        out = [answers]
        if isinstance(aliases, list):
            out.extend(str(x) for x in aliases if x)
        return list(dict.fromkeys(out))
    if isinstance(answers, list):
        out: list[str] = []
        for item in answers:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict) and "text" in item:
                out.append(str(item["text"]))
        if isinstance(aliases, list):
            out.extend(str(x) for x in aliases if x)
        return list(dict.fromkeys(out))
    return []


def _normalize_passages(record: dict[str, Any]) -> list[Passage]:
    raw = record.get("passages")
    if raw is None:
        raw = record.get("contexts", record.get("context", record.get("paragraphs", [])))
    passages: list[Passage] = []
    if isinstance(raw, dict):
        raw = list(raw.values())
    for idx, item in enumerate(raw or []):
        if isinstance(item, dict):
            pid = str(item.get("id", item.get("pid", item.get("idx", item.get("title", f"p{idx}")))))
            title = str(item.get("title", item.get("heading", "")))
            text = str(item.get("text", item.get("paragraph_text", item.get("content", ""))))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            title = str(item[0])
            sentences = item[1]
            text = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
            pid = title or f"p{idx}"
        else:
            pid = f"p{idx}"
            title = ""
            text = str(item)
        if text.strip() or title.strip():
            passages.append(Passage(id=pid, title=title, text=text))
    return passages


def _gold_ids(record: dict[str, Any], passages: list[Passage]) -> set[str]:
    explicit = record.get("gold_passage_ids", record.get("support_passage_ids", []))
    if explicit:
        return {str(x) for x in explicit}
    support = record.get("supporting_facts", record.get("supports", []))
    title_to_id = {p.title: p.id for p in passages if p.title}
    idx_to_id = {p.id: p.id for p in passages}
    ids: set[str] = set()
    for item in support or []:
        if isinstance(item, str):
            ids.add(title_to_id.get(item, item))
        elif isinstance(item, (list, tuple)) and item:
            ids.add(title_to_id.get(str(item[0]), str(item[0])))
        elif isinstance(item, dict):
            val = item.get("passage_id", item.get("title", item.get("pid")))
            if val is not None:
                ids.add(title_to_id.get(str(val), str(val)))
    raw = record.get("passages")
    if raw is None:
        raw = record.get("contexts", record.get("context", record.get("paragraphs", [])))
    if isinstance(raw, dict):
        raw = list(raw.values())
    for idx, item in enumerate(raw or []):
        if isinstance(item, dict) and item.get("is_supporting") is True:
            val = item.get("id", item.get("pid", item.get("idx", item.get("title", idx))))
            ids.add(title_to_id.get(str(val), idx_to_id.get(str(val), str(val))))
    for step in record.get("question_decomposition", []) or []:
        if isinstance(step, dict):
            val = step.get("paragraph_support_idx")
            if val is not None:
                ids.add(idx_to_id.get(str(val), str(val)))
    return ids


def normalize_record(record: dict[str, Any], idx: int) -> QAExample:
    passages = _normalize_passages(record)
    qid = str(record.get("id", record.get("_id", f"ex{idx}")))
    question = str(record.get("question", record.get("query", "")))
    answers = _as_answers(record)
    gold_ids = _gold_ids(record, passages)
    return QAExample(
        id=qid,
        question=question,
        answers=answers,
        passages=passages,
        gold_passage_ids=gold_ids,
        supporting_facts=record.get("supporting_facts", []),
        metadata={k: v for k, v in record.items() if k not in {"passages", "contexts", "context", "paragraphs"}},
    )


def load_examples(dataset_name: str, dataset_cfg: dict[str, Any]) -> list[QAExample]:
    loader = dataset_cfg.get("loader", "jsonl")
    limit = dataset_cfg.get("limit")
    if loader == "jsonl":
        path = dataset_cfg.get("path")
        if not path or not Path(path).exists():
            raise FileNotFoundError(
                f"Dataset {dataset_name} path not found: {path}. "
                "Set datasets.<name>.path in config.yaml to a normalized JSONL file."
            )
        records = read_jsonl(path)
    elif loader == "hf":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install `datasets` to use loader: hf") from exc
        name = dataset_cfg["hf_name"]
        subset = dataset_cfg.get("hf_config")
        split = dataset_cfg.get("split", "validation")
        ds = load_dataset(name, subset, split=split) if subset else load_dataset(name, split=split)
        records = [dict(x) for x in ds]
    else:
        raise ValueError(f"Unsupported loader: {loader}")
    if dataset_cfg.get("answerable_only"):
        records = [r for r in records if r.get("answerable", True) is True]
    if limit:
        records = records[: int(limit)]
    examples = [normalize_record(r, i) for i, r in enumerate(records)]
    return [ex for ex in examples if ex.question and ex.passages]
