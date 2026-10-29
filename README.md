# Contract Intelligence Assistant (simple Python version)

A beginner-friendly contract retrieval project for the JioStar take-home assignment. It ranks passages using a lightweight hybrid of word-level and character-level TF-IDF, preserves source character offsets, and abstains when every result is below a configurable threshold. The implementation is intentionally local and does not require an LLM.

## 1. Requirements
- Python 3.10 or newer
- Internet access to download the public benchmark
- Windows, macOS, or Linux

## 2. Install
Open a terminal in this project folder:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\\Scripts\\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Then:
```bash
pip install -r requirements.txt
```

## 3. Download the dataset
The easiest way is to run the included downloader. It fetches only the CUAD benchmark and CUAD contract files from the official dataset mirror (not the other benchmark sources):

```bash
pip install huggingface_hub
python download_data.py
python -m src.validate_data
```

Dataset source: https://huggingface.co/datasets/awinml/legalbench-rag

The downloader places files in this layout:

```text
data/
  raw/
    benchmarks/
      cuad.json
    corpus/
      cuad/
        ... contract text files ...
```

The loader selects exactly 194 CUAD mini queries using the sampling method in the assignment and loads only contract files referenced by their gold snippets. CUAD has 4,042 full queries; do not use all of them for this assignment. The dataset's exact file paths can vary between download methods; if the loader reports missing files, preserve the dataset's original relative paths under `data/raw/` and inspect `resolve_contract_path()` in `src/data_loader.py`.

**Offset check:** Before trusting evaluation, verify that `contract_text[start:end]` matches the answer text from each benchmark snippet. Dataset file encoding and newline conversion can affect offsets.

## 4. Run the web app
```bash
streamlit run app.py
```

## 5. Run the CLI
```bash
python cli.py Does this agreement allow sublicensing
```

Or:
```bash
python cli.py
```

## 6. Run tests
```bash
python -m pytest -q
```

The included unit tests check chunk offsets, relevant-passage ranking, and the keyword baseline. Passing these tests does not mean the benchmark has been downloaded or full evaluation has passed.

## 7. Evaluate the baseline
```bash
python -m src.evaluate --split train --method tfidf
```

After selecting a final configuration using training data, run the test evaluation once for final reporting:
```bash
python -m src.evaluate --split test --method tfidf
```

Test output is saved under `reports/`. Do not use test results to tune parameters.

## 8. Improvement loop
Run:
```bash
python -m src.improve
```

It runs the baseline plus five parameter/method experiments using the training split and writes `reports/experiment_log.csv`. Compare recall, precision, hit rate, and abstention together. The script logs experiments; review the metrics and document which configuration is accepted or rejected. Do not claim a change improved performance unless the measured numbers support it.

## 9. Unanswerable questions
The 30 starter questions are in `data/unanswerable/unanswerable.json`. They are candidate questions about topics that may not be covered by the corpus. You must manually verify them against the indexed contracts before treating them as labelled unanswerable examples. The split is derived from the exact question text using the assignment's split function.

Before starting the improvement loop, commit the final unanswerable set:
```bash
git add data/unanswerable/unanswerable.json
git commit -m "Add fixed unanswerable evaluation questions"
```
Do not edit the set after that commit. Count the train/test items and report the counts.

## 10. Project structure
```text
app.py                       Streamlit interface
cli.py                       Command-line interface
src/data_loader.py           Dataset loading and required split
src/retriever.py             Chunking, TF-IDF and keyword retrieval
src/pipeline.py              Shared setup
src/evaluate.py              Character-span and abstention metrics
src/improve.py               Baseline + five training experiments
src/config.py                Default settings
tests/test_retriever.py      Core unit tests
data/unanswerable/           Fixed unanswerable question set
configs/                     Baseline and final configuration examples
reports/                     Evaluation report and generated logs
```

## 11. Simple explanation of the code
1. **Chunking:** splits a long contract into smaller passages and keeps their character offsets.
2. **TF-IDF:** turns text into numbers based on words and how informative they are.
3. **Cosine similarity:** compares the question vector with each passage vector.
4. **Ranking:** shows the passages with the highest similarity.
5. **Abstention:** if every result is below the threshold, returns no passage.
6. **Evaluation:** compares retrieved character ranges against expert-provided answer spans.
7. **Improvement loop:** tests alternative chunk sizes, overlap, retrieval method, and threshold on train only.

## 12. GitHub submission checklist
1. Run `python -m pytest -q`.
2. Run `python -m src.validate_data` and resolve any offset/path mismatch before trusting metrics.
3. Run the training evaluation and `python -m src.improve`; inspect the generated JSON/CSV.
4. Commit the fixed unanswerable set before starting experiments, as required by the assignment.
5. Once the final settings are fixed, run the test evaluation exactly once and record the number of test runs in `reports/REPORT.md`.
6. Fill `reports/REPORT.md` with actual metrics and at least 10 real failure examples. Never leave placeholder values or invent results.
7. Push the repository to a public GitHub repository. Keep `data/raw/` out of Git because the dataset is downloaded separately; include setup instructions.

## 13. Assumptions and limitations
- This is a retrieval assistant, not a legal adviser.
- The threshold is a practical baseline and must be tuned on train.
- The supplied unanswerable queries require manual validation against the downloaded corpus.
- The assignment expects at least 10 concrete failure analyses; add them after evaluating real outputs.
- AI tool usage: ChatGPT was used to help scaffold and explain the code. Review, run, test, and understand every file before submitting.
- This project is a starting implementation. Validate the benchmark schema, path mapping, offset consistency, and metric definitions against the dataset documentation before treating the results as final.
