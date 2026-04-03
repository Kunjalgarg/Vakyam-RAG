import os
from src.rag_system import PDFRAG

MODEL = "models/meta-llama-3-8b-instruct.Q4_K_M.gguf"
PDF = "data/raw/kafan.pdf"

rag = PDFRAG(MODEL)

if not os.path.exists("data/processed/index.faiss"):
    print("🔧 First time setup...")
    rag.process_pdf(PDF)
else:
    print("⚡ Loading index...")
    rag.load()

print("\n🚀 Ready!\n")

while True:
    q = input("Ask (or exit): ")

    if q.lower() in ["exit", "quit"]:
        break

    ans, src = rag.answer(q)

    print("\nANSWER:\n", ans)
    print("\nSources:", src)