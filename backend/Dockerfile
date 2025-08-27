FROM python:3.11-slim
WORKDIR /app

# dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instala dependencias Python
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# Variables/puerto para Cloud Run
ENV PORT=8080
EXPOSE 8080

# Arranque del API
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
