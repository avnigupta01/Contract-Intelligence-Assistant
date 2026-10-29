
"""Download the CUAD benchmark and its contract files directly."""
from pathlib import Path
from huggingface_hub import snapshot_download

project_root = Path(__file__).resolve().parent
data_root = project_root / "data" / "raw"
data_root.mkdir(parents=True, exist_ok=True)

print("Downloading CUAD dataset. Please wait...")

snapshot_download(
    repo_id="awinml/legalbench-rag",
    repo_type="dataset",
    allow_patterns=[
        "benchmarks/cuad.json",
        "corpus/cuad/*.txt",
    ],
    local_dir=str(data_root),
)

benchmark = data_root / "benchmarks" / "cuad.json"
corpus = data_root / "corpus" / "cuad"
contracts = list(corpus.glob("*.txt")) if corpus.exists() else []

if not benchmark.exists():
    raise FileNotFoundError(f"Benchmark not found: {benchmark}")

if not contracts:
    raise FileNotFoundError(f"No contract files found in: {corpus}")

print("\nDownload complete!")
print(f"Benchmark: {benchmark}")
print(f"Contract files downloaded: {len(contracts)}")
print("\nNext, run: python -m src.validate_data")
