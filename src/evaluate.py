"""Evaluate retrieval using character-level overlap and abstention."""
import json
from pathlib import Path
from src.pipeline import build_system
from src.data_loader import assign_split
from src import config

OUT_DIR = Path("reports")
OUT_DIR.mkdir(exist_ok=True)


def merge_intervals(intervals):
    intervals = sorted(intervals)
    merged = []
    for start, end in intervals:
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def intersection_length(a, b):
    """Length of intersection between two lists of non-overlapping intervals."""
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def score_query(results, gold):
    """Score character overlap. A result only matches gold from the same document."""
    if not gold:
        return {"precision": 0.0, "recall": 0.0, "hit": 0}

    gold_by_file = {}
    for item in gold:
        gold_by_file.setdefault(item["file_path"], []).append((item["start"], item["end"]))
    gold_by_file = {name: merge_intervals(spans) for name, spans in gold_by_file.items()}

    retrieved_by_file = {}
    for item in results:
        retrieved_by_file.setdefault(item["file_path"], []).append((item["start"], item["end"]))
    retrieved_by_file = {name: merge_intervals(spans) for name, spans in retrieved_by_file.items()}

    retrieved_chars = sum(end - start for spans in retrieved_by_file.values() for start, end in spans)
    gold_chars = sum(end - start for spans in gold_by_file.values() for start, end in spans)
    overlap = 0
    for name, gold_spans in gold_by_file.items():
        overlap += intersection_length(gold_spans, retrieved_by_file.get(name, []))

    return {
        "precision": overlap / max(1, retrieved_chars),
        "recall": overlap / max(1, gold_chars),
        "hit": int(overlap > 0)
    }


def load_unanswerable():
    path = Path("data/unanswerable/unanswerable.json")
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def run_evaluation(method=None, chunk_size=None, overlap=None, split="train", threshold=None):
    rows, _, retriever = build_system(method, chunk_size, overlap)
    threshold = config.MIN_SCORE if threshold is None else threshold
    chosen = [row for row in rows if row["split"] == split]
    result_by_k = {}
    failures = []

    for k in [1, 3, 5, 10]:
        scores = []
        for row in chosen:
            results = retriever.search(row["query"], top_k=k, min_score=threshold)
            score = score_query(results, row["gold"])
            scores.append(score)
            if k == 5 and score["recall"] < 0.5:
                failures.append({
                    "query": row["query"],
                    "gold": row["gold"],
                    "returned": results[:2],
                    "reason": "Less than 50% of gold answer characters were covered."
                })
        count = len(scores)
        result_by_k[str(k)] = {
            "character_precision": sum(s["precision"] for s in scores) / max(1, count),
            "character_recall": sum(s["recall"] for s in scores) / max(1, count),
            "hit_rate": sum(s["hit"] for s in scores) / max(1, count)
        }

    unanswerable = [item for item in load_unanswerable()
                    if assign_split(item["query"]) == split]
    correct_abstentions = sum(
        not retriever.search(item["query"], top_k=5, min_score=threshold)
        for item in unanswerable
    )

    return {
        "split": split,
        "method": retriever.method,
        "threshold": threshold,
        "query_count": len(chosen),
        "metrics_at_k": result_by_k,
        "unanswerable_count": len(unanswerable),
        "correct_abstentions": correct_abstentions,
        "unanswerable_abstention_rate": correct_abstentions / max(1, len(unanswerable)),
        "failures_at_5": failures
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--method", choices=["tfidf", "keyword"], default="tfidf")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    result = run_evaluation(method=args.method, split=args.split, threshold=args.threshold)
    print(json.dumps(result, indent=2))
    output = OUT_DIR / f"evaluation_{args.split}_{args.method}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {output}")
