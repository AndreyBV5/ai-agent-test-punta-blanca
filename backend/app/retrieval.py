# app/retrieval.py
import os
from typing import List, Tuple
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.docstore.document import Document

load_dotenv()

PINECONE_API_KEY   = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX     = os.environ["PINECONE_INDEX"]                 # p.ej. "punta-blanca"
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")   # el mismo que usaste en ingest
INTEGRATED_MODEL   = os.getenv("INTEGRATED_MODEL", "multilingual-e5-large")
RAG_TOP_K          = int(os.getenv("RAG_TOP_K", "8"))             # subimos a 8

_pc = None
_index = None

def _ensure_index():
    global _pc, _index
    if _index is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
        _index = _pc.Index(PINECONE_INDEX)
    return _pc, _index

def load_vectorstore():
    # compat: devolvemos el índice
    _, index = _ensure_index()
    return index

def build_retriever(vs):
    # compat (no se usa con Pinecone)
    return vs

_KEYWORD_HINTS = {
    "ceo": ["ceo", "chief executive officer", "fundador", "founder", "equipo", "team", "about"],
    "cto": ["cto", "chief technology officer", "fundador", "equipo", "team", "about"],
    "coo": ["coo", "chief operating officer", "fundador", "equipo", "team", "about"],
    "servicio": ["services", "service", "servicios", "oferta", "our services"],
}

def _needs_about_boost(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ["ceo", "cto", "coo", "founder", "fundador", "equipo", "team", "about"])

def _needs_services_boost(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ["servicio", "services", "oferta"])

def similarity_with_scores(vs, query: str) -> List[Tuple[Document, float]]:
    pc, index = _ensure_index()

    # 1) vector del query con E5 en modo "query"
    resp = pc.inference.embed(
        model=INTEGRATED_MODEL,
        inputs=[query],
        parameters={"input_type": "query", "truncate": "END"},
    )
    qvec = resp.data[0].values

    # 2) búsqueda principal
    res = index.query(
        vector=qvec,
        top_k=RAG_TOP_K,
        include_metadata=True,
        namespace=PINECONE_NAMESPACE,
    )

    matches = list(res.matches or [])

    # 3) BOOST heurístico: si la pregunta parece de “equipo/fundadores” añade
    #   algunos candidatos del about; si es de servicios, añade de /services.
    def _boost_by_source(substrs: list[str], extra_k: int = 4):
        nonlocal matches
        # consulta “vacía” por vector + filtro semántico aproximado
        more = index.query(
            vector=qvec,
            top_k=extra_k,
            include_metadata=True,
            namespace=PINECONE_NAMESPACE,
        ).matches or []
        # filtra por URL y concatena sin duplicados
        src_seen = {m.id for m in matches}
        for m in more:
            src = (m.metadata or {}).get("source", "").lower()
            if any(s in src for s in substrs) and m.id not in src_seen:
                matches.append(m)
                src_seen.add(m.id)

    if _needs_about_boost(query):
        _boost_by_source(["about-us", "/es/about-us", "nosotros", "about"])
    if _needs_services_boost(query):
        _boost_by_source(["/services", "/es/services", "services-tech-details"])

    # 4) normaliza salida a (Document, score)
    out: List[Tuple[Document, float]] = []
    for m in matches[:RAG_TOP_K]:  # recorta al K final
        meta = m.metadata or {}
        text = meta.get("page_content") or meta.get("text") or meta.get("content") or ""
        # Si por alguna razón viniera vacío, al menos coloca la URL para que el LLM tenga ancla
        if not text:
            text = f"(ver fuente: {meta.get('source', '')})"
        out.append((Document(page_content=text, metadata=meta), float(m.score)))
    return out
