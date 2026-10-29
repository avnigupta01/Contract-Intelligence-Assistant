"""Load the CUAD mini benchmark and the contracts it refers to."""
import json
import random
from pathlib import Path

DATA_DIR = Path("data")
BENCHMARK_PATH = DATA_DIR / "raw" / "benchmarks" / "cuad.json"
CORPUS_DIR = DATA_DIR / "raw" / "corpus" / "cuad"


def select_mini(tests, n=194):
    """Use the deterministic sampling procedure specified in the assignment."""
    if len(tests) <= n:
        return tests

    def key(test):
        snippets = test.get("snippets", [])
        file_path = snippets[0].get("file_path", "") if snippets else ""
        random.seed(file_path)
        return random.random()

    return sorted(tests, key=key)[:n]


def assign_split(query):
    """Use the fixed train/test split from the assignment."""
    import hashlib
    value = int(hashlib.sha256(query.strip().encode("utf-8")).hexdigest(), 16)
    return "test" if value % 10 < 3 else "train"


def find_value(obj, possible_names, default=None):
    """Return the first available field name from a dictionary."""
    for name in possible_names:
        if name in obj:
            return obj[name]
    return default


def load_benchmark():
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {BENCHMARK_PATH}. Follow the README dataset download steps."
        )
    raw = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    tests = select_mini(raw.get("tests", []))
    rows = []

    for test in tests:
        query = find_value(test, ["query", "question", "text"], "")
        snippets = test.get("snippets", [])
        gold = []
        for snippet in snippets:
            file_path = find_value(snippet, ["file_path", "document", "doc_id"], "")
            span = snippet.get("span")
            start = find_value(snippet, ["start", "start_char", "char_start"])
            end = find_value(snippet, ["end", "end_char", "char_end"])
            if span and len(span) == 2:
                start, end = span[0], span[1]
            answer = find_value(snippet, ["answer", "text", "snippet"], "")
            if file_path and start is not None and end is not None:
                gold.append({
                    "file_path": str(file_path),
                    "start": int(start),
                    "end": int(end),
                    "answer": str(answer)
                })
        if query:
            rows.append({
                "query": str(query),
                "gold": gold,
                "split": assign_split(str(query))
            })
    return rows


def resolve_contract_path(file_path):
    """Try common paths used by the dataset mirror."""
    path = Path(file_path)
    candidates = [
        path if path.is_absolute() else Path.cwd() / path,
        CORPUS_DIR / path.name,
        CORPUS_DIR / path,
        DATA_DIR / "raw" / "corpus" / path,
        DATA_DIR / "raw" / file_path,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def load_contracts(rows):
    """Load only the contracts referenced by the selected 194 queries."""
    wanted = set()
    for row in rows:
        for gold in row["gold"]:
            wanted.add(gold["file_path"])

    contracts = []
    missing = []
    for file_path in sorted(wanted):
        path = resolve_contract_path(file_path)
        if path is None:
            missing.append(file_path)
            continue
        text = path.read_text(encoding="utf-8")
        contracts.append({"file_path": file_path, "text": text})

    if missing:
        print(f"Warning: could not find {len(missing)} contract files.")
        print("Example missing paths:", missing[:3])
    if not contracts:
        raise RuntimeError(
            "No contracts were loaded. Check that corpus/cuad files are in data/raw/corpus/cuad/."
        )
    return contracts
