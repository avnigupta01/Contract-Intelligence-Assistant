"""Simple boundary-aware hybrid TF-IDF retrieval with source offsets."""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src import config


def make_chunks(text, size=None, overlap=None):
    """Split text near natural boundaries while preserving exact character offsets."""
    size = config.CHUNK_SIZE if size is None else size
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    if size <= 0 or overlap < 0 or size <= overlap:
        raise ValueError("Require CHUNK_SIZE > CHUNK_OVERLAP >= 0.")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            lower = start + size // 2
            candidates = []
            for marker in ("\n\n", "\n", ". ", "; ", ": "):
                pos = text.rfind(marker, lower, end)
                if pos >= lower:
                    candidates.append((pos + len(marker), marker))
            if candidates:
                end = max(candidates, key=lambda item: item[0])[0]
        raw = text[start:end]
        passage = raw.strip()
        if passage:
            left_trim = len(raw) - len(raw.lstrip())
            actual_start = start + left_trim
            chunks.append({
                "text": passage, "file_path": "",
                "start": actual_start, "end": actual_start + len(passage)
            })
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


class ClauseRetriever:
    """Ranks passages using a lightweight word/character TF-IDF ensemble."""
    def __init__(self, contracts, method=None, chunk_size=None, overlap=None):
        self.method = method or config.RETRIEVAL_METHOD
        self.chunks = []
        for contract in contracts:
            for chunk in make_chunks(contract["text"], chunk_size, overlap):
                chunk["file_path"] = contract["file_path"]
                self.chunks.append(chunk)
        if not self.chunks:
            raise ValueError("No text chunks were created.")

        self.word_vectorizer = self.char_vectorizer = None
        self.word_matrix = self.char_matrix = None
        if self.method == "tfidf":
            texts = [chunk["text"] for chunk in self.chunks]
            self.word_vectorizer = TfidfVectorizer(
                lowercase=True, stop_words="english", ngram_range=(1, 2),
                sublinear_tf=True, max_features=200000
            )
            self.word_matrix = self.word_vectorizer.fit_transform(texts)
            self.char_vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                max_features=100000, min_df=1
            )
            self.char_matrix = self.char_vectorizer.fit_transform(texts)
        elif self.method != "keyword":
            raise ValueError(f"Unsupported retrieval method: {self.method}")

    def search(self, query, top_k=None, min_score=None):
        top_k = config.TOP_K if top_k is None else top_k
        min_score = config.MIN_SCORE if min_score is None else min_score
        if top_k <= 0:
            return []
        query = query.strip()
        if not query:
            return []

        if self.method == "tfidf":
            word_query = self.word_vectorizer.transform([query])
            char_query = self.char_vectorizer.transform([query])
            word_scores = cosine_similarity(word_query, self.word_matrix).ravel()
            char_scores = cosine_similarity(char_query, self.char_matrix).ravel()
            # Word similarity carries semantic-term overlap; character n-grams
            # help with legal terminology, plurals, and minor spelling variation.
            scores = 0.8 * word_scores + 0.2 * char_scores
        else:
            terms = set(re.findall(r"\b\w+\b", query.lower()))
            scores = np.array([
                len(terms & set(re.findall(r"\b\w+\b", c["text"].lower()))) /
                max(1, len(terms)) for c in self.chunks
            ], dtype=float)

        ranked_ids = np.argsort(scores)[::-1]
        results = []
        for index in ranked_ids:
            score = float(scores[index])
            if score < min_score:
                break
            chunk = self.chunks[index]
            results.append({
                "text": chunk["text"], "file_path": chunk["file_path"],
                "start": chunk["start"], "end": chunk["end"], "score": score
            })
            if len(results) >= top_k:
                break
        return results
