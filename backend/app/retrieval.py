import os
from typing import List, Tuple
from langchain.docstore.document import Document
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
# ⚠️ Debe coincidir EXACTO con el nombre del índice que creaste en Pinecone (dim=1024)
PINECONE_INDEX   = os.environ["PINECONE_INDEX"]              # p. ej. "punta-blanca-1024"
# Modelo integrado de Pinecone Inference para embeddings (dim 1024)
INTEGRATED_MODEL = os.getenv("INTEGRATED_MODEL", "multilingual-e5-large")
RAG_TOP_K        = int(os.getenv("RAG_TOP_K", "4"))

_pc = None
_index = None

def _ensure_index():
    """Singleton del cliente e índice Pinecone."""
    global _pc, _index
    if _index is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
        _index = _pc.Index(PINECONE_INDEX)
    return _pc, _index

def load_vectorstore():
    """Compat con el grafo: devolvemos el índice de Pinecone."""
    _, index = _ensure_index()
    return index

def build_retriever(vs):
    """Compat; no se usa con Pinecone directamente."""
    return vs

def similarity_with_scores(vs, query: str) -> List[Tuple[Document, float]]:
    """
    1) Embebe el query con Pinecone Inference (input_type='query') usando E5 (1024 dim).
    2) Hace query al índice.
    3) Devuelve [(Document, score)].
    """
    pc, index = _ensure_index()

    # 1) Embedding del query
    resp = pc.inference.embed(
        model=INTEGRATED_MODEL,
        inputs=[query],
        parameters={"input_type": "query", "truncate": "END"},
    )
    qvec = resp.data[0].values  # 1024 floats

    # 2) Búsqueda en Pinecone
    res = index.query(vector=qvec, top_k=RAG_TOP_K, include_metadata=True)

    out: List[Tuple[Document, float]] = []
    for m in res.matches:
        meta = m.metadata or {}
        # En el upsert guardamos el texto bajo "page_content"
        text = meta.get("page_content") or meta.get("text") or meta.get("content") or ""
        doc  = Document(page_content=text, metadata=meta)
        out.append((doc, float(m.score)))
    return out
