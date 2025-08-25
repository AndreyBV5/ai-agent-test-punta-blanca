# app/retrieval.py
import os
from typing import List, Tuple
from langchain.docstore.document import Document
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX   = os.environ["PINECONE_INDEX"]              # p.ej. "punta-blanca"
INTEGRATED_MODEL = os.getenv("INTEGRATED_MODEL", "multilingual-e5-large")
RAG_TOP_K        = int(os.getenv("RAG_TOP_K", "4"))

_pc = None
_index = None

def _ensure_index():
    """Obtiene singletons de cliente e índice Pinecone."""
    global _pc, _index
    if _index is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
        _index = _pc.Index(PINECONE_INDEX)
    return _pc, _index

def load_vectorstore():
    """Compat con tu grafo: antes devolvías un vectorstore; ahora devolvemos el índice Pinecone."""
    _, index = _ensure_index()
    return index

def build_retriever(vs):
    """Compat; no se usa con Pinecone directamente."""
    return vs

def similarity_with_scores(vs, query: str) -> List[Tuple[Document, float]]:
    """
    Embebe el query con el modelo integrado E5 (1024 dim) y consulta Pinecone.
    Devuelve [(Document, score)].
    """
    pc, index = _ensure_index()

    # 1) Embedding del query con Pinecone Inference (tipo 'query' para E5)
    resp = pc.inference.embed(
        model=INTEGRATED_MODEL,
        inputs=[query],
        parameters={"input_type": "query", "truncate": "END"},
    )
    qvec = resp.data[0].values  # 1024 floats

    # 2) Búsqueda en Pinecone
    res = index.query(
        vector=qvec,
        top_k=RAG_TOP_K,
        include_metadata=True
    )

    out: List[Tuple[Document, float]] = []
    for m in res.matches:
        meta = m.metadata or {}
        # Si en el upsert guardaste el texto del chunk:
        text = meta.get("page_content") or meta.get("text") or meta.get("content") or ""
        doc  = Document(page_content=text, metadata=meta)
        out.append((doc, float(m.score)))
    return out
