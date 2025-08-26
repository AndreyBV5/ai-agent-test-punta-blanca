# app/retrieval.py
import os
from typing import List, Tuple
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.docstore.document import Document

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX   = os.environ["PINECONE_INDEX"]
# Si está vacío/no definido => namespace __default__ (None en SDK)
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "").strip()

INTEGRATED_MODEL   = os.getenv("INTEGRATED_MODEL", "multilingual-e5-large")
RAG_TOP_K          = int(os.getenv("RAG_TOP_K", "4"))

DENY_SUBSTR = ["privacy", "terms", "cookies", "aviso-legal", "legal", "policy"]
KEYS_DEF = ["qué es", "que es", "what is", "somos", "es una empresa", "empresa", "inteligencia artificial", "ai"]

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
    """Para compat con el resto del agente, devolvemos el índice Pinecone."""
    _, index = _ensure_index()
    return index

def _needs_about_boost(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ["ceo", "cto", "coo", "founder", "fundador", "equipo", "team", "about", "qué es", "que es", "what is"])

def _needs_services_boost(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ["servicio", "services", "oferta"])

def _is_boilerplate(src: str) -> bool:
    s = (src or "").lower()
    return any(k in s for k in DENY_SUBSTR)

def _def_score(text: str) -> int:
    t = (text or "").lower()
    return sum(1 for k in KEYS_DEF if k in t)

def similarity_with_scores(vs, query: str) -> List[Tuple[Document, float]]:
    """
    1) Embebe el query con E5 (input_type='query')
    2) Hace query a Pinecone (namespace solo si está definido)
    3) Aplica filtros/boosts
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
        "top_k": max(RAG_TOP_K, 10),  # pilla más candidatos para reordenar
        "include_metadata": True,
    }
    if PINECONE_NAMESPACE:
        q_kwargs["namespace"] = PINECONE_NAMESPACE

    res = index.query(**q_kwargs)
    matches = list(res.matches or [])

    # 3a) Relega boilerplate (privacy/terms) al final
    matches = [m for m in matches if not _is_boilerplate((m.metadata or {}).get("source", ""))] + \
              [m for m in matches if _is_boilerplate((m.metadata or {}).get("source", ""))]

    # 3b) BOOST por about/services/linkedin
    def _boost_by_source(substrs: list[str], extra_k: int = 8):
        more = index.query(**{**q_kwargs, "top_k": extra_k}).matches or []
        seen = {m.id for m in matches}
        for m in more:
            src = (m.metadata or {}).get("source", "").lower()
            if any(s in src for s in substrs) and m.id not in seen:
                matches.append(m)
                seen.add(m.id)

    if _needs_about_boost(query):
        _boost_by_source(["/about", "/about-us", "/es/about", "/es/about-us", "linkedin", "company/puntablancasolutions"])
    if _needs_services_boost(query):
        _boost_by_source(["/services", "/es/services", "services-tech-details"])

    # 3c) Reordenado final por "definición" + score semántico
    tmp = []
    for m in matches:
        meta = m.metadata or {}
        text = meta.get("page_content") or meta.get("text") or meta.get("content") or ""
        tmp.append((m, text, _def_score(text)))
    tmp.sort(key=lambda x: (x[2], x[0].score), reverse=True)

    # 4) Normaliza a [(Document, score)] limitado a top_k
    out: List[Tuple[Document, float]] = []
    for m, text, _ in tmp[:RAG_TOP_K]:
        meta = dict(m.metadata or {})
        if "page_content" not in meta:
            meta["page_content"] = text
        out.append((Document(page_content=text, metadata=meta), float(m.score)))
    return out
