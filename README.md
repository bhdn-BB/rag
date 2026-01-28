# 🤖 Agentic RAG Application

A production-ready Retrieval-Augmented Generation (RAG) system with agent capabilities, built with FastAPI and LangChain.

## 📋 Features

- **Vector Memory Management**: Store and search documents using semantic embeddings
- **Agentic RAG**: Intelligent question-answering with context retrieval
- **Multiple Document Sources**: Support for file uploads and URL ingestion
- **Advanced Search**: Bi-encoder + cross-encoder reranking
- **Conversation Memory**: Stateful agent interactions
- **Docker Support**: Easy deployment with Docker Compose

---

## 🚀 Quick Start

### 1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd rag_app
```

### 2️⃣ Install Dependencies

**Option A: Using pip (local development)**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Tip:** Regenerate requirements.txt from your venv:
```bash
pip freeze > requirements.txt
```

### 3️⃣ Configure Environment

Create your `.env` file:
```bash
cp .env.example .env
```

Fill in your API keys and model configurations:
```env
LLM_API_KEY=your_llm_api_key_here
EMBEDDING_MODEL=your_embedding_model
RERANK_MODEL=your_rerank_model
SUMMARIZATION_MODEL=your_summarization_model
```

### 4️⃣ Run with Docker

```bash
docker-compose up --build
```

**Access the application:**
- 📚 **FastAPI Docs**: http://localhost:8000/docs
- 🎨 **Streamlit UI**: http://localhost:8501

**Rebuild without cache (if needed):**
```bash
docker-compose build --no-cache
docker-compose up
```

---

## 🔌 API Endpoints

### 📁 Vector Memory Management

Manage your document store and perform semantic search.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/vector-memory/documents/file` | Upload and index a file |
| `POST` | `/vector-memory/documents/url` | Add documents from URL |
| `POST` | `/vector-memory/search` | Search documents semantically |
| `DELETE` | `/vector-memory/delete` | Delete documents by metadata |
| `DELETE` | `/vector-memory/clear` | Clear entire vector store |
| `GET` | `/vector-memory/status` | Get vector store statistics |

---

### 💬 RAG Agent Chat

Interact with the intelligent RAG agent for question-answering.

| Method | Path         | Description |
|--------|--------------|-------------|
| `POST` | `/rag/query` | Ask a question to the RAG agent |
| `POST` | `/rag/reset` | Reset conversation memory |

#### Examples

**Ask a question:**
```bash
curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain the main concepts from the uploaded documents"
  }'
```

**Response format:**
```json
{
  "answer": "Based on the documents...",
  "sources": ["doc1.pdf", "doc2.pdf"],
  "docs_count": 3,
  "error": null
}
```
---

## 🏗️ Architecture

### Components

- **Vector Memory**: Manages document storage with ChromaDB
- **Embedders**: 
  - `HFBiEmbedder`: Bi-encoder for fast retrieval
  - `HFCrossEncoder`: Cross-encoder for precise reranking
- **Document Parser**: `DBNParser` handles multiple file formats
- **RAG Agent**: Stateful agent with conversation memory
---

## 🛠️ Development

### Run Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run Streamlit (in another terminal)
streamlit run streamlit_app.py
```

---

## 📝 Supported File Formats

- PDF documents
- Word documents (.docx)
- Web pages (via URL)

---

## 🐛 Troubleshooting

### Common Issues

**1. Port already in use:**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

**2. Docker build fails:**
```bash
# Clean Docker cache
docker system prune -a
docker-compose build --no-cache
```

**3. Vector store not persisting:**
- Check volume mounts in `docker-compose.yml`
- Ensure proper permissions on data directories
