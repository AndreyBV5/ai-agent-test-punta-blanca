# from .retrieval import load_vectorstore, similarity_with_scores  # <- quita esto
from .retrieval_pinecone import get_store, similarity_with_scores

_store = None
def _ensure_store():
    global _store
    if _store is None:
        _store = get_store()
    return _store

def retrieval_node(state: AgentState):
    store = _ensure_store()
    pairs = similarity_with_scores(store, state["question"])
    docs = [d for d, _ in pairs]
    scores = [float(s) for _, s in pairs]
    sources = list(dict.fromkeys([d.metadata.get("source", "") for d in docs if d]))
    return {**state, "docs": docs, "scores": scores, "sources": sources}
