# 📘 README – Agente RAG para Punta Blanca Solutions

Este proyecto implementa un **Agente RAG (Retrieval Augmented Generation)** que responde preguntas sobre **Punta Blanca Solutions**, utilizando **LangChain**, **Gemini (Google Generative AI)** y **Pinecone** como vectorstore.  

La información de contexto proviene de:  
- Scraping del sitio oficial [https://www.puntablanca.ai](https://www.puntablanca.ai)  
- Un archivo de texto con datos de LinkedIn (`data/sources/linkedin_punta_blanca.txt`)

---

## 🗂️ Estructura del proyecto (carpeta `backend/`)

```
backend/
├─ app/
│  ├─ __init__.py
│  ├─ main.py              # FastAPI (endpoints / y /api/ask)
│  ├─ graph.py             # Orquesta nodos (input -> retrieval -> generation)
│  ├─ generation.py        # Prompt + llamada a Gemini
│  ├─ retrieval.py         # Consulta a Pinecone (E5 query + boosts + filtros)
│  └─ schemas.py           # Pydantic models (AskRequest/AskResponse)
├─ ingest/
│  └─ build_vectorstore_pinecone.py  # Crawler + embeddings + upsert Pinecone
├─ data/
│  └─ sources/
│     └─ linkedin_punta_blanca.txt   # Texto base de LinkedIn
├─ requirements.txt
└─ Dockerfile
```

## 🔧 1. Arquitectura del sistema

El proyecto sigue un flujo **end-to-end** de tipo **RAG (Retrieval Augmented Generation)**:

1. **Usuario hace una pregunta** a través del endpoint `/api/ask`.  
2. **Embeddings de la query**: la pregunta se convierte en un vector usando el modelo `multilingual-e5-large`.  
3. **Búsqueda en Pinecone**: se consulta el índice `punta-blanca` para recuperar los fragmentos más relevantes (chunks del sitio web y de LinkedIn).  
4. **Contexto**: los documentos recuperados se formatean y se pasan al modelo generativo.  
5. **Generación de respuesta**: Gemini (`gemini-1.5-flash`) recibe la pregunta + contexto y devuelve una respuesta natural en el idioma del usuario.  
6. **Respuesta al cliente**: FastAPI entrega un JSON con `answer`, `sources` (URLs) y `confidence` (nivel de similitud promedio).  

**Componentes principales:**
- `app/main.py` → API FastAPI con endpoints (`/` y `/api/ask`).  
- `graph.py` → define el grafo de nodos: input → retrieval → generation → output.  
- `retrieval.py` → conecta con Pinecone y aplica boosts/filtros.  
- `generation.py` → construye prompt y llama al LLM de Gemini.  
- `ingest/build_vectorstore_pinecone.py` → crawler + embeddings + upsert en Pinecone.  
- `data/sources/linkedin_punta_blanca.txt` → fuente manual adicional.  

---

## 🚀 2. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/ai-agent-test.git
cd ai-agent-test
```

---

## 🐍 3. Crear y activar entorno virtual

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux/MacOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 📦 4. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

---

## 🔑 5. Configuración de variables de entorno

Crea un archivo `.env` en la raíz del proyecto con tus credenciales:

```env
GOOGLE_API_KEY=tu_api_key_google
GENERATION_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=text-embedding-004

PINECONE_API_KEY=tu_api_key_pinecone
PINECONE_INDEX=punta-blanca #O el nombre que se le ponga
PINECONE_HOST=https://url_que_pinecone_proporcione
INTEGRATED_MODEL=multilingual-e5-large
RAG_TOP_K=4
```

> 🔎 **Notas**  
> - Tu API Key de Pinecone debe comenzar con `pcsk_...`  
> - El índice debe estar creado en la región `us-east-1` (si no, el script lo crea automáticamente).

---

## 🗂️ 6. Ingesta de datos (crawling + embeddings)

⚠️ Este paso es **obligatorio** la primera vez (y cada vez que quieras refrescar la información).

Desde la carpeta `backend`:

```bash
cd backend
python ingest/build_vectorstore_pinecone.py
```

Deberías ver algo así:

```
[crawl] Recolectando páginas…
[chunks] 60
[embed] E5(passages) …
[upsert] 60 vectores → Pinecone (ns='__default__')
[ok] Índice actualizado.
```

Esto significa que ya tienes datos en Pinecone y el agente podrá usarlos.

---

## ▶️ 7. Levantar el servidor local (FastAPI + Uvicorn)

```bash
cd backend
uvicorn app.main:app --reload --port 8080
```

Visita en el navegador:  
👉 [http://localhost:8080](http://localhost:8080)

Prueba un POST con curl o Postman:

```bash
curl -X POST "http://localhost:8080/api/ask" \
  -H "content-type: application/json" \
  -d "{\"question\":\"¿Qué es Punta Blanca?\"}"
```

Respuesta esperada:
```json
{
  "answer": "Punta Blanca Solutions es una empresa costarricense dedicada al desarrollo de soluciones de Inteligencia Artificial...",
  "sources": ["https://www.linkedin.com/company/puntablancasolutions/", "https://www.puntablanca.ai/about-us"],
  "confidence": 0.89
}
```

---

## ☁️ 8. Deployment en Google Cloud Run (CI/CD con GitHub)

> Sin YAML. Todo por interfaz gráfica en <https://console.cloud.google.com/run>.

1. **Sube tu código al repositorio** (GitHub o similar).  
2. Entra a **Cloud Run** → **Crear servicio**.
3. **Región**: `us-central1` (o la que prefieras).
4. **Tipo de despliegue**: **Implementar desde código fuente**.  
   - Conecta tu repositorio (GitHub) si te lo pide.  
   - Selecciona el repo y la carpeta raíz del servicio (normalmente el repo completo).  
   - Cloud Build detectará el `Dockerfile` o usará Buildpacks (en caso de no hacerlo selecciona opcion usar dockefile y poner la ruta y el nombre del archivo).
5. **Nombre del servicio**: `ai-agent` (o el que elijas).
6. **Autenticación**: Permitir **no autenticado** (si quieres endpoint público).
7. **Configuración del contenedor**:
   - **Puerto**: `8080`
   - **Ruta de comprobación de estado**: `/`  (tu raíz responde JSON)
   - **CPU/Memoria**: 1 vCPU / 512MB o 1GB (ajústalo a tu gusto)
   - **Concurrencia**: 80 (valor por defecto está bien)
8. **Variables de entorno** (pestaña *Variables y secretos*):
   - `GOOGLE_API_KEY=tu_api`
   - `EMBEDDING_MODEL=text-embedding-004`
   - `GENERATION_MODEL=gemini-1.5-flash`
   - `PINECONE_API_KEY=tu_api`
   - `PINECONE_INDEX=punta-blanca`
   - `INTEGRATED_MODEL=multilingual-e5-large`
   - `RAG_TOP_K=4`
   - `PINECONE_HOST=url_que_proporcione_al_crear_index_pinecone`
   - (no definas `PINECONE_NAMESPACE`)
9. Clic en **Crear** y espera a que finalice el build + deploy.
10. Copia la **URL del servicio** (forma `https://<servicio>-<hash>-<región>.run.app`).

**Probar en producción**:
```bash
curl -X POST "https://<tu-servicio>.run.app/api/ask"   -H "content-type: application/json"   -d "{"question":"¿Qué es Punta Blanca?"}"
```

**Logs**: Cloud Run → tu servicio → **Registros**.  
En los logs verás líneas como `[retrieved sources] [...]` para depurar el retrieval.

---

## 🔄 9) Actualizar contenido (re-ingesta)

Cuando cambien las fuentes (web/LinkedIn), **vuelve a ejecutar** localmente:

```bash
cd backend
python ingest/build_vectorstore_pinecone.py
```

No necesitas redeploy de Cloud Run para que lea lo nuevo (el servicio consulta Pinecone en cada petición).

---


## 🌐 10. Uso de Pinecone (Dashboard Web)

Esta sección explica cómo **crear tu cuenta, API key e índice en Pinecone** para usarlo como **vectorstore**.  
👉 Si no quieres hacerlo manualmente, el script de ingesta (`backend/ingest/build_vectorstore_pinecone.py`) **crea el índice automáticamente** si no existe (usando los valores de tu `.env`).

### A) Crear cuenta y API Key

1. Ve a [https://app.pinecone.io](https://app.pinecone.io) y crea una cuenta (plan Starter sirve).  
2. En el menú izquierdo, entra a **API Keys → Create API key**.  
3. Ponle un nombre (ejemplo: `ai-agent-local`) y crea la key.  
4. Copia la key (formato `pcsk_...`) y agrégala a tu `.env`:

### B) Crear el índice (Serverless)

Aunque el script puede crearlo automáticamente, si prefieres hacerlo desde la interfaz web:

1. En el menú izquierdo, ve a **Indexes → Create Index**.  
2. Configura así:  
   - **Name**: `punta-blanca` *(debe coincidir con `PINECONE_INDEX`)*  
   - **Deployment**: `Serverless`  
   - **Cloud**: `AWS`  
   - **Region**: `us-east-1`  
   - **Metric**: `cosine`  
   - **Dimension**: `1024` ← requerido por el modelo `multilingual-e5-large`  
3. Haz clic en **Create Index**.  

💡 **Dimensiones según modelo de embeddings**:  
- `multilingual-e5-large` ⇒ **1024**  
- `multilingual-e5-base` ⇒ 768  
- `e5-small` ⇒ 384  

> Ajusta también el valor de `INTEGRATED_MODEL` en tu `.env`.

---

### C) Namespaces

- Un índice puede tener múltiples **namespaces** (particiones lógicas).  
- Si no defines `PINECONE_NAMESPACE`, el SDK usa `__default__`.  
- En este proyecto se recomienda dejar el namespace **vacío** para evitar confusiones.  

---

## 📐 11. Decisiones técnicas

- **FastAPI**: framework rápido y con documentación automática.  
- **LangChain**: simplifica la orquestación de RAG y el manejo de prompts.  
- **Gemini (Google Generative AI)**: modelo multilingüe con alto rendimiento en español.  
- **Pinecone**: vectorstore administrado en la nube, confiable y escalable.  
- **Modelo `multilingual-e5-large` (1024 dims)**: equilibrio entre precisión semántica y costo.  
- **`RAG_TOP_K=6`**: número de pasajes recuperados, buscando balance entre cobertura y ruido.  
- **Namespace vacío (`__default__`)**: evita confusiones y mantiene datos centralizados.  
- **Boosting en retrieval**: prioriza páginas relevantes (`about`, `services`, LinkedIn`) frente a páginas poco útiles (`privacy`, `terms`).  
- **Arquitectura desacoplada**: la ingesta se corre solo para actualizar Pinecone, mientras que el servicio responde siempre en tiempo real consultando esa base vectorial.  

---

## 🧾 11. Resumen de flujo de trabajo

1. Clonar repo y crear venv.  
2. Instalar dependencias.  
3. Configurar `.env` con API keys.  
4. Ejecutar ingesta (`build_vectorstore_pinecone.py`).  
5. Levantar servidor local con `uvicorn`. (En caso de querer probarlo local)
6. Levantar servidor con`GCP` para desplegar en Cloud Run. (Probarlo en producción)
7. Hacer preguntas vía API.  
8. Si agregas nuevas fuentes → volver a correr ingesta.  





