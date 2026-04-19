import chromadb
from data.documents import docs
from rag.embeddings import load_embedding_model

def create_vector_db():
    model = load_embedding_model()

    documents = [doc["text"] for doc in docs]
    ids = [doc["id"] for doc in docs]
    metadatas = [{"topic": doc["topic"]} for doc in docs]

    embeddings = model.encode(documents).tolist()

    client = chromadb.Client()
    collection = client.create_collection(name="legal_kb")

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    return collection, model

import re

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks


import hashlib

def add_uploaded_doc(collection, model, text):
    chunks = chunk_text(text)

    embeddings = model.encode(chunks).tolist()

    # 🔥 deduplication via file hash
    file_id = hashlib.md5(text.encode()).hexdigest()

    ids = [f"{file_id}_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"topic": "User Upload"}]*len(chunks)
    )