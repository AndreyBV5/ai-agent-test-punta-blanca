# app/retrieval.py
import os
from typing import List, Tuple
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.docstore.document import Document

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX   = os.environ["PINECONE_INDEX"]
# 👇 Namespace OPCIONAL. Si está vacío/no definido, usaremos __default__ (None en el SDK).
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "").strip()

INTEGRATED_MODEL   = os.getenv("INTEGRATED_MODEL", "multilingual-e5-large")
RAG_TOP_K          = int(os.getenv("RAG_TOP_K", "4"))

_pc = None
_index = None


def _ensure_index():
    """Singleton de cliente e índice Pinecone."""
    global _pc, _index
    if _index is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
        _index = _pc.Index(PINECONE_INDEX)
    return _pc, _index


def load_vectorstore():
    """Compat: devolvemos el índice Pinecone."""
    _, index = _ensure_index()
    return index


def build_retriever(vs):
    """Compat (no se usa con Pinecone)."""
    return vs


def _needs_about_boost(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in [
        "ceo", "cto", "coo", "founder", "fundador", "equipo", "team", "about"
    ])


def _needs_services_boost(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ["servicio", "services", "oferta"])


def similarity_with_scores(vs, query: str) -> List[Tuple[Document, float]]:
    """
    1) Embebe el query con E5 (input_type='query')
    2) Hace query a Pinecone (namespace solo si está definido)
    3) Aplica un pequeño boost heurístico para about/services
    4) Devuelve [(Document, score)]
    """
    pc, index = _ensure_index()

    # 1) vector del query
    resp = pc.inference.embed(
        model=INTEGRATED_MODEL,
        inputs=[query],
        parameters={"input_type": "query", "truncate": "END"},
    )
    qvec = resp.data[0].values

    # 2) query principal
    q_kwargs = {
        "vector": qvec,
        "top_k": RAG_TOP_K,
        "include_metadata": True,
    }
    # Namespace solo si fue configurado. None => __default__
    if PINECONE_NAMESPACE:
        q_kwargs["namespace"] = PINECONE_NAMESPACE

    res = index.query(**q_kwargs)
    matches = list(res.matches or [])

    # 3) BOOST heurístico (misma query args)
    def _boost_by_source(substrs: list[str], extra_k: int = 6):
        more = index.query(**{**q_kwargs, "top_k": extra_k}).matches or []
        seen = {m.id for m in matches}
        for m in more:
            src = (m.metadata or {}).get("source", "").lower()
            if any(s in src for s in substrs) and m.id not in seen:
                matches.append(m)
                seen.add(m.id)

    if _needs_about_boost(query):
        _boost_by_source(["about-us", "/es/about-us", "nosotros", "about"])
    if _needs_services_boost(query):
        _boost_by_source(["/services", "/es/services", "services-tech-details"])

    # 4) Normaliza a [(Document, score)]
    out: List[Tuple[Document, float]] = []
    for m in matches[:RAG_TOP_K]:
        meta = m.metadata or {}
        text = (
            meta.get("page_content")
            or meta.get("text")
            or meta.get("content")
            or f"(ver fuente: {meta.get('source', '')})"
        )
        out.append((Document(page_content=text, metadata=meta), float(m.score)))
    return out
