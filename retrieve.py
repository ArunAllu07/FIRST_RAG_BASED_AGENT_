import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(
    path="./chroma_db"
)

# Load existing collection
collection = client.get_collection(
    name="surface_notes"
)


def search_pdf(query, top_k=3):
    # Convert query into embedding
    query_embedding = model.encode([query]).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    return results["documents"][0]


# Ask user a question
query = input("Ask something about the PDF: ")

results = search_pdf(query)

print("\n--- RELEVANT INFORMATION ---")

for i, result in enumerate(results):
    print(f"\nResult {i + 1}:")
    print(result)