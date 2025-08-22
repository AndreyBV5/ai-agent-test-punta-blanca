FROM python:3.11-slim
WORKDIR /app

# Dependencias del sistema (faiss necesita BLAS en algunos entornos)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# Puerto estándar de FastAPI en Cloud Run
ENV PORT=8080
EXPOSE 8080

# En el primer arranque, si no hay índice, intenta construirlo (requiere GOOGLE_API_KEY)
CMD python ingest/build_vectorstore.py || true && \
    uvicorn app.main:app --host 0.0.0.0 --port ${PORT}