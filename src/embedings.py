import os
import pickle
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# === CONFIG ===
PDF_PATH = "D:\Students\Vakyam-RAG\data\kafan.pdf"
EMBEDDING_FILE = "embeddings.pkl"

# === LOAD PDF ===
reader = PdfReader(PDF_PATH)
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

# === SPLIT TEXT ===
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_text(text)

print(f"Total chunks: {len(chunks)}")

# === LOAD EMBEDDING MODEL ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === CREATE EMBEDDINGS ===
embeddings = model.encode(chunks, show_progress_bar=True)

# === SAVE TO FILE ===
with open(EMBEDDING_FILE, "wb") as f:
    pickle.dump({
        "chunks": chunks,
        "embeddings": embeddings
    }, f)

print("✅ Embeddings saved!")