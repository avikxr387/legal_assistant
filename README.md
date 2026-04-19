# Legal Document Assistant (RAG + LangGraph)

## Overview

This project implements a legal document assistant capable of answering questions from domain-specific documents using Retrieval-Augmented Generation (RAG). The system integrates structured reasoning via LangGraph, semantic retrieval using ChromaDB, and memory-aware conversations.

The assistant supports both preloaded knowledge and dynamically uploaded documents, enabling users to query legal content efficiently.

---

## Key Features

* Document-based question answering using RAG
* Support for document upload (TXT, PDF, PPTX)
* Semantic retrieval using vector embeddings
* Multi-turn conversation with persistent memory
* Deterministic tool support for precise computations
* Self-evaluation with retry mechanism for answer quality
* Source attribution for retrieved answers
* Streamlit-based chat interface

---

## System Architecture

The system is built using a graph-based execution model.

```
User Input
   ↓
Memory Node
   ↓
Router Node
   ↓
[Retrieve] / [Tool] / [Skip]
   ↓
Answer Node
   ↓
Evaluation Node
   ↓
Save Node
```

### Components

* **StateGraph (LangGraph)**: Controls flow between nodes
* **ChromaDB**: Stores embeddings for retrieval
* **SentenceTransformer**: Generates embeddings (`all-MiniLM-L6-v2`)
* **Groq LLM**: Generates responses
* **MemorySaver**: Maintains conversation history
* **Tool Node**: Handles deterministic operations

---

## Project Structure

```
legal-assistant/
│
├── data/
│   └── documents.py          # Static knowledge base
│
├── rag/
│   ├── embeddings.py         # Embedding model loader
│   └── vectordb.py           # Vector DB + document ingestion
│
├── graph/
│   ├── state.py              # State schema
│   ├── nodes.py              # Node implementations
│   ├── graph_builder.py      # Graph assembly
│   └── test_nodes.py         # Node testing
│
├── capstone_streamlit.py     # Frontend application
├── test_app.py               # End-to-end testing
├── ragas_eval.py             # Evaluation pipeline
│
└── README.md
```

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/your-username/legal-assistant.git
cd legal-assistant
```

---

### 2. Create a virtual environment

```
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install dependencies

```
pip install streamlit langgraph chromadb sentence-transformers groq pypdf python-pptx
```

---

## Environment Configuration

Set the Groq API key.

### Windows

```
setx GROQ_API_KEY "your_api_key_here"
```

Restart the terminal after setting the variable.

---

## Running the Application

```
streamlit run capstone_streamlit.py
```

---

## Document Upload Pipeline

The system supports dynamic ingestion of user documents.

### Supported Formats

* TXT
* PDF
* PPTX

### Processing Flow

```
Upload → Text Extraction → Chunking → Embedding → ChromaDB Storage
```

Uploaded content is merged into the existing vector database and becomes immediately queryable.

---

## Query Types

### Domain Queries

* What is breach of contract?
* Explain termination clause
* What damages can be claimed?

### Document-Based Queries

* What does clause 3 state?
* Summarize the uploaded document

### Tool-Based Queries

* Count consonants in the document
* What is the current date and time?

---

## Evaluation

The system includes a self-evaluation mechanism:

* Faithfulness scoring (0 to 1)
* Retry logic for low-confidence answers
* Source-based grounding enforcement

Evaluation can be performed using:

```
python ragas_eval.py
```

---

## Design Decisions

* Separation of routing and execution using LangGraph
* Static + dynamic knowledge integration
* Deterministic tools for exact computations
* Sliding window memory to control token usage
* Context-grounded prompting to reduce hallucination

---

## Limitations

* No persistent database (in-memory ChromaDB)
* No authentication or access control
* No handling for very large documents
* Retrieval is purely semantic (no keyword search)

---

## Future Improvements

* Persistent vector database
* Hybrid retrieval (BM25 + embeddings)
* Cross-encoder reranking
* Document summarization pipeline
* Multi-document reasoning
* Async processing for large uploads

---

## Conclusion

This project demonstrates a complete RAG-based system with memory, tool integration, and evaluation mechanisms. It provides a structured approach to building domain-specific assistants capable of handling both static and dynamic knowledge sources.


# Author

**AVIK HALDER**
GitHub: [https://github.com/avikxr387](https://github.com/avikxr387)
