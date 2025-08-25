# ingest/build_vectorstore_pinecone.py
import os, re, uuid
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX   = os.environ["PINECONE_INDEX"]                # tu índice integrado (1024)
INTEGRATED_MODEL = os.getenv("INTEGRATED_MODEL", "multilingual-e5-large")

DOMAIN        = "https://www.puntablanca.ai/"
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", "100"))
MAX_EMBED_BATCH = int(os.getenv("MAX_EMBED_BATCH", "64"))      # para no saturar la API

def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        t.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

def crawl(domain: str, max_pages: int = 30) -> list[Document]:
    seen, queue, docs = set(), [domain], []
    client = httpx.Client(timeout=20, follow_redirects=True)
    try:
        while queue and len(seen) < max_pages:
            url = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)
            try:
                r = client.get(url)
                if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
                    continue
                text = clean_text(r.text)
                if text:
                    docs.append(Document(page_content=text, metadata={"source": url}))
                soup = BeautifulSoup(r.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = urljoin(url, a["href"])
                    if urlparse(href).netloc == urlparse(domain).netloc:
                        href = href.split("#")[0]
                        if href not in seen and href not in queue:
                            queue.append(href)
            except Exception:
                continue
    finally:
        client.close()
    return docs

def load_linkedin_txt() -> list[Document]:
    path = os.path.join(os.path.dirname(__file__), "..", "data", "sources", "linkedin_punta_blanca.txt")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    return [Document(page_content=txt, metadata={"source": "linkedin"})] if txt else []

def dedup_by_source_and_head(docs: list[Document]) -> list[Document]:
    seen, out = set(), []
    for d in docs:
        key = (d.metadata.get("source", ""), d.page_content[:60])
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out

def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def main():
    print("[crawl] Obteniendo datos de puntablanca.ai ...")
    docs = crawl(DOMAIN, max_pages=30)
    docs += load_linkedin_txt()
    docs = dedup_by_source_and_head(docs)
    if not docs:
        raise SystemExit("No hay documentos para indexar.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    texts  = [c.page_content for c in chunks]
    metas  = [c.metadata for c in chunks]

    pc = Pinecone(api_key=PINECONE_API_KEY)

    # (Opcional) crear índice si no existe (1024). Si ya lo creaste en consola, esto no corre.
    existing = [i["name"] for i in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        print(f"[pinecone] Creando índice {PINECONE_INDEX} (dim=1024, integrated={INTEGRATED_MODEL})...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    # ---------- Embeddings con Pinecone Inference ----------
    print(f"[embed] Solicitando embeddings ({INTEGRATED_MODEL}) vía Pinecone Inference...")
    all_vectors = []
    for batch_texts in batched(texts, MAX_EMBED_BATCH):
        # ¡Aquí está el parámetro que faltaba!
        resp = pc.inference.embed(
            model=INTEGRATED_MODEL,
            inputs=batch_texts,
            parameters={
                "input_type": "passage",   # documentos/pasajes (para upsert)
                "truncate": "END"
            },
        )
        all_vectors.extend([e.values for e in resp.data])  # 1024 floats cada uno

    index = pc.Index(PINECONE_INDEX)

    print(f"[pinecone] Subiendo {len(all_vectors)} chunks a Pinecone en lotes de {BATCH_SIZE}...")
    batch = []
    for vec, meta in zip(all_vectors, metas):
        batch.append({"id": str(uuid.uuid4()), "values": vec, "metadata": meta})
        if len(batch) >= BATCH_SIZE:
            index.upsert(vectors=batch)
            batch = []
    if batch:
        index.upsert(vectors=batch)

    print("[ok] Upsert completado 👍")

if __name__ == "__main__":
    main()
