"""Check the selected CUAD mini benchmark and gold character offsets."""
from src.data_loader import load_benchmark, load_contracts

def main():
    rows = load_benchmark()
    contracts = load_contracts(rows)
    by_path = {c["file_path"]: c["text"] for c in contracts}
    total = 0
    valid = 0
    missing = 0
    for row in rows:
        for gold in row["gold"]:
            total += 1
            text = by_path.get(gold["file_path"])
            if text is None:
                missing += 1
                continue
            if text[gold["start"]:gold["end"]] == gold["answer"]:
                valid += 1
            else:
                print("Offset mismatch:", row["query"], gold["file_path"], gold["start"], gold["end"])
    print(f"Mini queries loaded: {len(rows)}")
    print(f"Contracts loaded: {len(contracts)}")
    print(f"Gold spans matching exactly: {valid}/{total}")
    print(f"Missing contract spans: {missing}")
    if valid != total or missing:
        raise SystemExit("Validation failed. Do not report evaluation metrics until this is fixed.")
    print("Dataset and character offsets validated.")

if __name__ == "__main__":
    main()
