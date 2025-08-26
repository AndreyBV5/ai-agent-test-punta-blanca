# app/retrieval.py
import os
from typing import List, Tuple
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.docstore.document import Document

load_dotenv()

PINECONE_API_KEY   = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX     = os.environ["PINECONE_INDEX"]                 # p.ej. "punta-blanca"
# MUY IMPORTANTE: cadena vacía "" => namespace __default__ en Pinecone
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
    """Compatibilidad: devolvemos el índice Pinecone."""
    _, index = _ensure_index()
    return index

def build_retriever(vs):
    """Compat (no se usa con Pinecone)."""
    return vs

def _query_index(index, qvec, top_k: int):
    """Envuelve index.query respetando el namespace solo si NO es vacío."""
    kwargs = dict(vector=qvec, top_k=top_k, include_metadata=True)
    if PINECONE_NAMESPACE:  # si está vacío => __default__ (no pasar)
        kwargs["namespace"] = PINECONE_NAMESPACE
    return index.query(**kwargs)

def similarity_with_scores(vs, query: str) -> List[Tuple[Document, float]]:
    """
    1) Embebe el query con E5 (input_type='query').
    2) Busca en Pinecone (y aplica un pequeño boost heurístico).
    3) Devuelve [(Document, score)].
    """
    pc, index = _ensure_index()

    # 1) vector del query (E5)
    resp = pc.inference.embed(
        model=INTEGRATED_MODEL,
        inputs=[query],
        parameters={"input_type": "query", "truncate": "END"},
    )
    qvec = resp.data[0].values  # 1024 floats

    # 2) búsqueda principal
    res = _query_index(index, qvec, RAG_TOP_K)
    matches = list(res.matches or [])

    # --- Boost heurístico opcional (about/services) ---
    ql = query.lower()

    def _boost_by_source(substrs: list[str], extra_k: int = 6):
        nonlocal matches
        more = _query_index(index, qvec, extra_k).matches or []
        seen = {m.id for m in matches}
        for m in more:
            src = (m.metadata or {}).get("source", "").lower()
            if any(s in src for s in substrs) and m.id not in seen:
                matches.append(m)
                seen.add(m.id)

    if any(k in ql for k in ["ceo", "cto", "coo", "founder", "fundador", "equipo", "team", "about"]):
        _boost_by_source(["/about-us", "/es/about-us", "about", "nosotros"])

    if any(k in ql for k in ["servicio", "services", "oferta", "our services"]):
        _boost_by_source(["/services", "/es/services", "services-tech-details"])

    # 3) normaliza salida a (Document, score)
    out: List[Tuple[Document, float]] = []
    for m in matches[:RAG_TOP_K]:
        meta = m.metadata or {}
        text = meta.get("page_content") or meta.get("text") or meta.get("content") or ""
        if not text:
            text = f"(ver fuente: {meta.get('source', '')})"
        out.append((Document(page_content=text, metadata=meta), float(m.score)))
    return out
