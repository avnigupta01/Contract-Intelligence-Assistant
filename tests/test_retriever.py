from src.retriever import make_chunks, ClauseRetriever


def test_chunks_keep_offsets():
    text = "First paragraph.\\n\\nSecond paragraph.\\n\\nThird paragraph."
    chunks = make_chunks(text, size=25, overlap=5)
    assert chunks
    for chunk in chunks:
        assert text[chunk["start"]:chunk["end"]].strip() == chunk["text"]


def test_retriever_finds_relevant_passage():
    contracts = [
        {"file_path": "a.txt", "text": "The licensee may sublicense the licensed content to approved third-party platforms."},
        {"file_path": "b.txt", "text": "Payment is due thirty days after receipt of invoice."}
    ]
    retriever = ClauseRetriever(contracts, method="tfidf", chunk_size=200, overlap=20)
    results = retriever.search("Can the content be sublicensed to third parties?", top_k=2, min_score=0.0)
    assert results
    assert results[0]["file_path"] == "a.txt"


def test_keyword_retriever_runs():
    contracts = [{"file_path": "a.txt", "text": "Exclusive rights last for five years."}]
    retriever = ClauseRetriever(contracts, method="keyword", chunk_size=100, overlap=10)
    assert retriever.search("exclusive rights", min_score=0.0)



def test_empty_query_returns_no_results():
    retriever = ClauseRetriever(
        [{"file_path": "a.txt", "text": "Confidential information must be protected."}],
        method="tfidf", chunk_size=100, overlap=10
    )
    assert retriever.search("   ") == []


def test_invalid_chunk_settings_raise():
    import pytest
    with pytest.raises(ValueError):
        make_chunks("sample text", size=10, overlap=10)
