# graph.py
from typing import List, TypedDict
from langgraph.graph import StateGraph, END
from langchain.schema import Document
from .retrieval import load_vectorstore, similarity_with_scores
from .generation import generate_answer

class AgentState(TypedDict):
    question: str
    docs: List[Document]
    scores: List[float]
    answer: str
    sources: List[str]
    confidence: float

def input_node(state: AgentState):
    q = state["question"].strip()
    if len(q) < 3:
        raise ValueError("Pregunta demasiado corta")
    return {**state, "question": q}

_VS = None
def _ensure_vs():
    global _VS
    if _VS is None:
        _VS = load_vectorstore()
    return _VS

def retrieval_node(state: AgentState):
    vs = _ensure_vs()
    pairs = similarity_with_scores(vs, state["question"])
    docs = [d for d, _ in pairs]
    scores = [float(s) for _, s in pairs]
    sources = list(dict.fromkeys([d.metadata.get("source", "") for d in docs if d]))
    return {**state, "docs": docs, "scores": scores, "sources": sources}

# IMPORTANT: async
async def generation_node(state: AgentState):
    answer = await generate_answer(state["question"], state.get("docs", []))
    scores = state.get("scores", [])
    conf = float(sum(scores)/len(scores)) if scores else 0.5
    return {**state, "answer": answer, "confidence": round(conf, 3)}

def output_node(state: AgentState):
    return {
        "answer": state.get("answer", ""),
        "sources": state.get("sources", []),
        "confidence": state.get("confidence", 0.0),
    }

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("input", input_node)
    g.add_node("retrieval", retrieval_node)
    g.add_node("generation", generation_node)  # <- async node
    g.add_node("output", output_node)

    g.set_entry_point("input")
    g.add_edge("input", "retrieval")
    g.add_edge("retrieval", "generation")
    g.add_edge("generation", "output")
    g.add_edge("output", END)
    return g.compile()
