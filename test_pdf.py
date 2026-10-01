from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_path = "documents/SURFACE.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    text += page.extract_text() or ""

print("Total characters:", len(text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- CHUNK {i + 1} ---")
    print(chunk)