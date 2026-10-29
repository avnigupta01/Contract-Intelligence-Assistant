"""Small one-command parameter sweep. It evaluates only the training split."""
import csv
from pathlib import Path
from src.evaluate import run_evaluation

# Six configurations = baseline + five attempts to improve it.
# Keep this list unchanged after you begin logging experiments.
CONFIGS = [
    {"name": "baseline", "method": "tfidf", "chunk_size": 1200, "overlap": 200, "threshold": 0.08},
    {"name": "iteration_1_smaller_chunks", "method": "tfidf", "chunk_size": 800, "overlap": 150, "threshold": 0.08},
    {"name": "iteration_2_larger_chunks", "method": "tfidf", "chunk_size": 1800, "overlap": 250, "threshold": 0.08},
    {"name": "iteration_3_more_overlap", "method": "tfidf", "chunk_size": 1200, "overlap": 400, "threshold": 0.08},
    {"name": "iteration_4_keyword_baseline", "method": "keyword", "chunk_size": 1200, "overlap": 200, "threshold": 0.08},
    {"name": "iteration_5_higher_threshold", "method": "tfidf", "chunk_size": 1200, "overlap": 200, "threshold": 0.15},
]


def main():
    path = Path("reports/experiment_log.csv")
    path.parent.mkdir(exist_ok=True)
    rows = []
    for config in CONFIGS:
        print("Running", config["name"])
        result = run_evaluation(
            method=config["method"],
            chunk_size=config["chunk_size"],
            overlap=config["overlap"],
            split="train",
            threshold=config["threshold"]
        )
        row = {
            **config,
            "query_count": result["query_count"],
             "precision_at_5": result["metrics_at_k"]["5"]["character_precision"],
            "recall_at_5": result["metrics_at_k"]["5"]["character_recall"],
            "hit_rate": result["metrics_at_k"]["5"]["hit_rate"],
            "unanswerable_count": result["unanswerable_count"],
            "abstention_rate": result["unanswerable_abstention_rate"],
            "decision": "REVIEW: compare recall, precision and abstention guardrails"
        }
        rows.append(row)
        print(row)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved training-only experiment results to {path}")
    print("This script logs results; inspect the guardrails and choose the final configuration manually.")
    print("Do not run this on test data or tune parameters using test results.")


if __name__ == "__main__":
    main()
