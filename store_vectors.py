from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# -----------------------------
# 1. Read PDF
# -----------------------------

pdf_path = "documents/SURFACE.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    text += page.extract_text() or ""

# -----------------------------
# 2. Split into chunks
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)

print("Number of chunks:", len(chunks))

# -----------------------------
# 3. Create embeddings
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks).tolist()

print("Embeddings created:", len(embeddings))

# -----------------------------
# 4. Create ChromaDB
# -----------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="surface_notes"
)

# -----------------------------
# 5. Store chunks
# -----------------------------

collection.upsert(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings
)

print("Successfully stored chunks in ChromaDB!")
print("Total documents:", collection.count())