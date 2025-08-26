import os
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document

load_dotenv()

GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-1.5-flash")

_SYSTEM = (
    "Eres un asistente de IA para preguntas sobre Punta Blanca Solutions. "
    "Usa SOLO la información del contexto recuperado. "
    "Si hay indicios parciales, redacta una respuesta breve y explícita sobre lo que SÍ aparece. "
    "Si realmente no hay nada útil, di que no aparece en el contexto. "
    "Responde en el idioma de la pregunta; sé conciso. "
    "Si la pregunta es sobre equipo directivo (CEO, CTO, COO, fundadores), busca explícitamente nombres y cargos en el contexto."
)

_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "Pregunta: {question}\n\nContexto:\n{context}\n\nResponde de forma clara."),
])

def _format_docs(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "")
        parts.append(f"[Fuente] {src}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

def _llm():
    return ChatGoogleGenerativeAI(
        model=GENERATION_MODEL,
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

def generate_answer(question: str, docs: List[Document]) -> str:
    prompt = _TEMPLATE.format_messages(
        question=question,
        context=_format_docs(docs),
    )
    resp = _llm().invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)
