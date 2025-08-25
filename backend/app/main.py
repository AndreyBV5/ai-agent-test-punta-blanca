from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import AskRequest, AskResponse
from .graph import build_graph

app = FastAPI(title="RAG Agent – Punta Blanca", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_graph = build_graph()

@app.get("/")
def root():
    return {"ok": True, "service": "rag-agent", "health": "green"}

@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        result = _graph.invoke({"question": req.question})
        return AskResponse(
            answer=str(result.get("answer", "")),
            sources=[str(s) for s in result.get("sources", [])],
            confidence=float(result.get("confidence", 0.0)),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
