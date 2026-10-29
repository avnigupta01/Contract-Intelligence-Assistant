import streamlit as st
from src.pipeline import build_system
from src import config

st.set_page_config(page_title="Contract Intelligence Assistant", page_icon="📄")
st.title("📄 Contract Intelligence Assistant")
st.write("Ask a question about the indexed contracts. Results include source and character offsets.")

@st.cache_resource
def load_system():
    return build_system()

try:
    rows, contracts, retriever = load_system()
    st.caption(f"Loaded {len(contracts)} contract files | Retrieval: {retriever.method}")
    question = st.text_input(
        "Your question",
        placeholder="Does this agreement allow sublicensing?"
    )
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=config.TOP_K)
    threshold = st.slider("Minimum relevance score", min_value=0.0, max_value=0.5, value=float(config.MIN_SCORE), step=0.01)

    if st.button("Search", type="primary") and question.strip():
        results = retriever.search(question, top_k=top_k, min_score=threshold)
        if not results:
            st.warning("No sufficiently relevant passage found in the indexed contracts.")
        else:
            for i, result in enumerate(results, start=1):
                st.subheader(f"{i}. Relevance score: {result['score']:.3f}")
                st.write(f"**Source:** `{result['file_path']}`")
                st.write(f"**Character offsets:** {result['start']}–{result['end']}")
                st.write(result["text"])
                st.divider()
except Exception as error:
    st.error(str(error))
    st.info("Check the dataset folder setup in README.md.")
