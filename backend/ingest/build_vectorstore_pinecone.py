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

BASE = "https://www.puntablanca.ai/"
SEEDS = [
    # Español
    "https://www.puntablanca.ai/es",
    "https://www.puntablanca.ai/es/about-us",
    "https://www.puntablanca.ai/es/contact-us",
    "https://www.puntablanca.ai/es/services-tech-details",
    # Inglés
    "https://www.puntablanca.ai/",
    "https://www.puntablanca.ai/about-us",
    "https://www.puntablanca.ai/contact-us",
    "https://www.puntablanca.ai/services",
]

SITEMAP   = urljoin(BASE, "sitemap.xml")
INDEX     = os.getenv("PINECONE_INDEX", "punta-blanca")
MODEL     = os.getenv("INTEGRATED_MODEL", "multilingual-e5-large")  # 1024 dims
# 👇 Namespace opcional: si está vacío, upsert a __default__ (namespace=None)
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "").strip()
MAX_PAGES = int(os.getenv("CRAWL_MAX_PAGES", "60"))

UA = {"User-Agent": "Mozilla/5.0 (pb-crawler)"}


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        t.decompose()
    txt = soup.get_text("\n")
    txt = re.sub(r"\n{2,}", "\n\n", txt)
    return txt.strip()


def try_sitemap():
    urls = []
    try:
        r = httpx.get(SITEMAP, timeout=15, headers=UA, follow_redirects=True)
        if r.status_code == 200 and "<urlset" in r.text:
            soup = BeautifulSoup(r.text, "xml")
            for loc in soup.find_all("loc"):
                u = loc.text.strip()
                if urlparse(u).netloc == urlparse(BASE).netloc:
                    urls.append(u.split("#")[0])
    except Exception:
        pass
    return list(dict.fromkeys(urls))


def crawl(max_pages=MAX_PAGES):
    seen, q, docs = set(), list(SEEDS + try_sitemap()), []
    client = httpx.Client(timeout=20, headers=UA, follow_redirects=True)
    while q and len(seen) < max_pages:
        url = q.pop(0)
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
                href = urljoin(url, a["href"]).split("?")[0].split("#")[0]
                if urlparse(href).netloc == urlparse(BASE).netloc:
                    if href not in seen and href not in q:
                        q.append(href)
        except Exception:
            continue
    client.close()
    return docs


def main():
    print("[crawl] Recolectando páginas…")
    docs = crawl()

    # Añade LinkedIn si existe
    linked = os.path.join(os.path.dirname(__file__), "..", "data", "sources", "linkedin_punta_blanca.txt")
    if os.path.exists(linked):
        with open(linked, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            if txt:
                docs.append(Document(page_content=txt, metadata={"source": "linkedin"}))

    if not docs:
        raise SystemExit("No se recolectó contenido.")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"[chunks] {len(chunks)}")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    # Crea índice si no existe (dim=1024 para multilingual-e5-large)
    existing = [i["name"] for i in pc.list_indexes()]
    if INDEX not in existing:
        print(f"[index] creando {INDEX} (dim=1024)…")
        pc.create_index(
            name=INDEX,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    idx = pc.Index(INDEX)

    # Embeddings PASSAGES
    texts = [c.page_content for c in chunks]
    metas = [c.metadata for c in chunks]
    print("[embed] E5(passages) …")
    emb = pc.inference.embed(
        model=MODEL,
        inputs=texts,
        parameters={"input_type": "passage", "truncate": "END"},
    )
    vecs = [d.values for d in emb.data]

    # Upsert (guarda SIEMPRE page_content + source)
    print(f"[upsert] {len(vecs)} vectores → Pinecone (ns='{NAMESPACE or '__default__'}')")
    batch, B = [], 100
    for v, meta, text in zip(vecs, metas, texts):
        meta = dict(meta or {})
        meta["page_content"] = text
        batch.append({"id": str(uuid.uuid4()), "values": v, "metadata": meta})
        if len(batch) >= B:
            idx.upsert(vectors=batch, namespace=(NAMESPACE or None))
            batch = []
    if batch:
        idx.upsert(vectors=batch, namespace=(NAMESPACE or None))
    print("[ok] Índice actualizado.")


if __name__ == "__main__":
    main()
