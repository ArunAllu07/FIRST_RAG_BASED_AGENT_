from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "Surface atoms have fewer neighbours and therefore higher surface energy.",
    "FCC bulk coordination number is 12.",
    "FCC (111) has the lowest surface energy."
]

embeddings = model.encode(texts)

print("Number of embeddings:", len(embeddings))
print("Embedding size:", len(embeddings[0]))
print("First embedding:")
print(embeddings[0])