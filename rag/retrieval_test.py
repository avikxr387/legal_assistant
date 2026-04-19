from rag.vectordb import create_vector_db

collection, model = create_vector_db()

# query = "What happens when a contract is broken?"
query = "Can I cancel a contract if the other party fails?"
query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

print("\nRetrieved Topics:")
for meta in results["metadatas"][0]:
    print("-", meta["topic"])

print("\nRetrieved Documents:")
for doc in results["documents"][0]:
    print("-", doc[:100], "...")