import os
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

VECTOR_PATH = os.getenv(
    "VECTOR_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "vectorstore"),
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))

def get_embeddings():
    # requiere GOOGLE_API_KEY en el entorno
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

def load_vectorstore() -> FAISS:
    embeddings = get_embeddings()
    vs = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
    return vs

def build_retriever(vs: FAISS):
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": RAG_TOP_K})

def similarity_with_scores(vs: FAISS, query: str) -> List[Tuple[Document, float]]:
    return vs.similarity_search_with_relevance_scores(query, k=RAG_TOP_K)
