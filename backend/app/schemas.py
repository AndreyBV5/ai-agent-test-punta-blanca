from pydantic import BaseModel, Field
from typing import List

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Pregunta del usuario")

class AskResponse(BaseModel):
    answer: str
    sources: List[str] = []
    confidence: float = 0.0