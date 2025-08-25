import os
from dotenv import load_dotenv
from pinecone import Pinecone

# carga el .env desde el backend
load_dotenv(dotenv_path="../backend/.env")

api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise RuntimeError("❌ Falta PINECONE_API_KEY en el entorno")

INDEX = os.getenv("PINECONE_INDEX", "punta-blanca")
MODEL = os.getenv("INTEGRATED_MODEL", "multilingual-e5-large")

pc = Pinecone(api_key=api_key)
idx = pc.Index(INDEX)

q = "¿Qué servicios ofrece Punta Blanca?"
emb = pc.inference.embed(
    model=MODEL, inputs=[q],
    parameters={"input_type": "query", "truncate": "END"}
)
vec = emb.data[0].values

res = idx.query(vector=vec, top_k=10, include_metadata=True)
for i, m in enumerate(res.matches, 1):
    print(i, m.score, m.metadata.get("source"))
