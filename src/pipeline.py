"""Shared setup used by the web app, CLI, and evaluation."""
from src.data_loader import load_benchmark, load_contracts
from src.retriever import ClauseRetriever


def build_system(method=None, chunk_size=None, overlap=None):
    rows = load_benchmark()
    contracts = load_contracts(rows)
    retriever = ClauseRetriever(contracts, method, chunk_size, overlap)
    return rows, contracts, retriever
