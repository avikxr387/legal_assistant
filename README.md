# Legal Document Assistant (RAG + LangGraph + FastAPI)

## Overview

This project implements a Legal Document Assistant capable of answering user queries from both preloaded knowledge and user-uploaded legal documents using a Retrieval-Augmented Generation (RAG) pipeline.

The system integrates semantic retrieval, structured reasoning, and large language models to provide accurate, context-grounded responses. It supports multi-turn conversations, document ingestion, and source-aware answering, making it suitable for legal document analysis and querying.

---

## Key Features

* Retrieval-Augmented Generation (RAG) for legal question answering
* Upload and query custom documents (TXT, PDF, PPTX)
* Semantic search using vector embeddings
* Multi-turn conversational memory
* Deterministic tool support (e.g., counting, date/time)
* Self-evaluation with faithfulness scoring
* Source attribution for answers
* Web-based chat interface (HTML, CSS, JavaScript)
* Drag-and-drop and button-based file upload

---

## System Architecture

The system uses a graph-based execution model powered by LangGraph.

User Query
→ Memory Node
→ Router Node
→ (Retrieve / Tool / Skip)
→ Answer Node
→ Evaluation Node
→ Response

---

## Core Components

### LangGraph

Manages structured flow between nodes such as routing, retrieval, answering, and evaluation.

### ChromaDB

Stores vector embeddings for semantic document retrieval.

### Sentence Transformers

Embedding model used:
all-MiniLM-L6-v2

### Groq LLM

Handles response generation using contextual inputs.

### Memory System

Maintains conversation state across multiple user queries.

### Tool Layer

Supports deterministic operations such as:

* Character counting
* Date/time retrieval

---

## Project Structure

legal-assistant/

data/
  documents.py

rag/
  embeddings.py
  vectordb.py

graph/
  state.py
  nodes.py
  graph_builder.py

backend/
  api.py

frontend/
  index.html

ragas_eval.py
test_app.py
check_env.py
README.md

---

## Installation

### 1. Clone Repository

git clone https://github.com/your-username/legal-assistant.git
cd legal-assistant

---

### 2. Create Virtual Environment

python -m venv venv
venv\Scripts\activate

---

### 3. Install Dependencies

pip install fastapi uvicorn langgraph chromadb sentence-transformers groq pypdf python-pptx duckduckgo-search

---

## Environment Setup

Set your Groq API key:

Windows:

setx GROQ_API_KEY "your_api_key_here"

Restart your terminal after setting the variable.

---

## Running the Application

### Start Backend

uvicorn backend.api:app --reload

Backend runs at:
http://127.0.0.1:8000

---

### Start Frontend

cd frontend
python -m http.server 5500

Open in browser:
http://localhost:5500

---

## Document Upload Pipeline

Upload → Text Extraction → Chunking → Embedding → ChromaDB Storage

Supported formats:

* TXT
* PDF
* PPTX

Uploaded documents are processed and become immediately available for querying.

---

## Supported Query Types

### Legal Knowledge Queries

* What is breach of contract?
* Explain termination clause

### Document-Based Queries

* What does clause 3 state?
* Summarize the uploaded document

### Tool-Based Queries

* Count consonants in the document
* What is the current date and time

---

## Evaluation System

The system includes an internal evaluation mechanism:

* Faithfulness scoring (0 to 1)
* Retry logic for low-confidence answers
* Context-grounded response validation

Run evaluation:

python ragas_eval.py

---

## Design Decisions

* Use of graph-based execution instead of linear pipelines
* Separation of routing and execution logic
* Integration of static and dynamic knowledge sources
* Deterministic tools for precise operations
* Context-grounded prompting to reduce hallucinations

---

## Limitations

* No persistent database (in-memory vector storage)
* No authentication or user management
* Limited handling of very large documents
* Retrieval based only on semantic similarity

---

## Future Improvements

* Persistent vector database (ChromaDB with storage)
* Hybrid retrieval (BM25 + embeddings)
* Cross-encoder re-ranking
* Document summarization pipeline
* Multi-document reasoning
* Backend-based chat history persistence
* Deployment with scalable infrastructure

---

## Conclusion

This project demonstrates a complete end-to-end RAG-based legal assistant integrating retrieval, reasoning, memory, and evaluation. It provides a structured and scalable foundation for building domain-specific AI assistants capable of handling both static and user-provided knowledge sources.

---

## Author

Avik Halder
GitHub: https://github.com/avikxr387
