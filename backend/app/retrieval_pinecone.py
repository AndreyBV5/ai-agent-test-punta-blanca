# # app/retrieval_pinecone.py
# import os
# from typing import List, Tuple
# from langchain.docstore.document import Document
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_pinecone import PineconeVectorStore

# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
# PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-puntablanca")
# RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))

# def get_store() -> PineconeVectorStore:
#     emb = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
#     return PineconeVectorStore(index_name=PINECONE_INDEX, embedding=emb)

# def similarity_with_scores(store: PineconeVectorStore, query: str) -> List[Tuple[Document, float]]:
#     # LangChain Pinecone devuelve (docs, scores) con similarity_search_with_score
#     results = store.similarity_search_with_score(query, k=RAG_TOP_K)
#     # normaliza a [(Document, float)]
#     return [(doc, float(score)) for doc, score in results]
