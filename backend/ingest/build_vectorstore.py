import os
import re
import json
import argparse
from urllib.parse import urljoin, urlparse
import httpx
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(BASE_DIR, "data", "sources")
VEC_DIR = os.path.join(BASE_DIR, "data", "vectorstore")
DOMAIN = "https://www.puntablanca.ai/"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

os.makedirs(SRC_DIR, exist_ok=True)
os.makedirs(VEC_DIR, exist_ok=True)

def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        t.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

def crawl(domain: str, max_pages: int = 20) -> list[Document]:
    seen, queue = set(), [domain]
    docs = []
    client = httpx.Client(timeout=20.0, follow_redirects=True)
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
            if not text:
                continue
            docs.append(Document(page_content=text, metadata={"source": url}))
            # descubrir enlaces internos
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"])
                if urlparse(href).netloc == urlparse(domain).netloc:
                    if "#" in href:
                        href = href.split("#")[0]
                    if href not in seen and href not in queue:
                        queue.append(href)
        except Exception:
            continue
    client.close()
    return docs

def load_linkedin_txt() -> list[Document]:
    path = os.path.join(SRC_DIR, "linkedin_punta_blanca.txt")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    if not txt:
        return []
    # Incluye la URL pública de LinkedIn si la conoces; si no, etiqueta como linkedin
    return [Document(page_content=txt, metadata={"source": "linkedin"})]

def build_index(docs: list[Document]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(VEC_DIR)
    print(f"[ok] Guardado FAISS en: {VEC_DIR} (chunks={len(chunks)})")

def main(do_crawl: bool):
    docs = []
    if do_crawl:
        print("[crawl] puntablanca.ai ...")
        docs.extend(crawl(DOMAIN, max_pages=30))
    docs.extend(load_linkedin_txt())

    if not docs:
        raise SystemExit("No hay documentos. Ejecuta con --crawl y/o agrega linkedin_punta_blanca.txt")

    build_index(docs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crawl", action="store_true", help="Crawlea puntablanca.ai y construye el índice")
    args = parser.parse_args()
    main(args.crawl)