import argparse
from src.pipeline import build_system
from src import config


def main():
    parser = argparse.ArgumentParser(description="Search contract clauses.")
    parser.add_argument("question", nargs="*", help="Question about a contract")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--threshold", type=float, default=config.MIN_SCORE)
    parser.add_argument("--method", choices=["tfidf", "keyword"], default=config.RETRIEVAL_METHOD)
    args = parser.parse_args()
    question = " ".join(args.question).strip()
    if not question:
        question = input("Enter your contract question: ").strip()
    if not question:
        print("Please enter a question.")
        return

    _, _, retriever = build_system(method=args.method)
    results = retriever.search(question, args.top_k, args.threshold)
    if not results:
        print("No sufficiently relevant passage found.")
        return
    for number, result in enumerate(results, 1):
        print(f"\\nResult {number} | score={result['score']:.3f}")
        print(f"Source: {result['file_path']}")
        print(f"Offsets: {result['start']}–{result['end']}")
        print(result["text"])


if __name__ == "__main__":
    main()
