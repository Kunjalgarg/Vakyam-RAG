import os
import faiss
import numpy as np
import difflib
import uuid
import psutil
import torch
import time

from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from pdf2image import convert_from_path
import pytesseract
from gtts import gTTS
import pygame


# =============================
# RAG SYSTEM
# =============================
class PDFRAG:

    def __init__(self, model_path, n_gpu_layers=0, n_batch=512):

        print("Loading LLM...")

        physical = psutil.cpu_count(logical=False)
        threads = max(1, physical - 1)

        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=threads,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )

        print("Loading embedding model...")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embedding_model = SentenceTransformer(
            "intfloat/multilingual-e5-base",
            device=device
        )

        print(f"Embeddings on: {device}")

        self.chunks = []
        self.index = None

        pygame.mixer.init()


    # -----------------------------
    # OCR
    # -----------------------------
    def extract_text(self, pdf_path):

        pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        os.environ["TESSDATA_PREFIX"] = r"C:\\Program Files\\Tesseract-OCR\\tessdata"
        poppler_path = r"C:\\poppler\\poppler-25.12.0\\Library\\bin"

        pages = convert_from_path(pdf_path, dpi=200, poppler_path=poppler_path)

        text = ""
        for page in pages:
            text += pytesseract.image_to_string(page, lang="hin") + "\n"

        self.full_text = self.clean_text(text)


    # -----------------------------
    # CHUNKING (BETTER)
    # -----------------------------
    def chunk_text(self, chunk_size=300, overlap=80):

        text = self.full_text

        self.chunks = [
            text[i:i + chunk_size]
            for i in range(0, len(text), chunk_size - overlap)
        ]

        print(f"✅ Total chunks: {len(self.chunks)}")


    # -----------------------------
    # CREATE INDEX
    # -----------------------------
    def create_index(self):

        print("Creating embeddings...")

        passages = ["passage: " + c for c in self.chunks]

        emb = self.embedding_model.encode(
            passages,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        dim = emb.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb)

        print("✅ FAISS index created")


    def clean_text(self, text):
        text = text.replace("\n", " ")
        text = " ".join(text.split())

        # fix spaced Hindi letters (basic)
        text = text.replace("म ा ध व", "माधव")
        text = text.replace("घ ी स ू", "घीसू")

        return text

    # -----------------------------
    # SAVE
    # -----------------------------
    def save(self, path="rag_data"):

        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        np.save(os.path.join(path, "chunks.npy"), self.chunks)

        print("✅ Index saved!")


    # -----------------------------
    # LOAD
    # -----------------------------
    def load(self, path="rag_data"):

        index_path = os.path.join(path, "index.faiss")
        chunk_path = os.path.join(path, "chunks.npy")

        if not os.path.exists(index_path) or not os.path.exists(chunk_path):
            raise FileNotFoundError("❌ Saved index not found. Run build first.")

        self.index = faiss.read_index(index_path)
        self.chunks = np.load(chunk_path, allow_pickle=True)

        print("✅ Index loaded!")


    # -----------------------------
    # QUERY FIX
    # -----------------------------
    def normalize_query(self, query):

        words = query.lower().split()

        dictionary = ["kafan", "ghisu", "madhav", "budiya"]

        fixed = []
        for w in words:
            match = difflib.get_close_matches(w, dictionary, n=1)
            fixed.append(match[0] if match else w)

        return " ".join(fixed)


    # -----------------------------
    # RETRIEVE (IMPROVED)
    # -----------------------------
    def retrieve(self, query, k=6):

        query = self.normalize_query(query)

        q = self.embedding_model.encode(
            ["query: " + query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, idx = self.index.search(q, k)

        context = []
        sources = []

        for i, score in zip(idx[0], scores[0]):
            if score > 0.15:  # 🔥 filter weak matches
                context.append(self.chunks[i])
                sources.append(f"Chunk {i}")

        # fallback if nothing passes filter
        if not context:
            for i in idx[0]:
                context.append(self.chunks[i])
                sources.append(f"Chunk {i}")

        return "\n".join(context), sources


    # -----------------------------
    # GENERATE
    # -----------------------------
    def generate(self, system, user):

        prompt = f"""<|start_header_id|>system<|end_header_id|>
{system}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

        out = self.llm(
            prompt,
            max_tokens=500,
            temperature=0.1,
            repeat_penalty=1.2,
            stop=["<|eot_id|>"]
        )

        return out["choices"][0]["text"].strip()


    # -----------------------------
    # ANSWER (BETTER PROMPTS)
    # -----------------------------
    def answer(self, q, lang="hindi"):

        context, src = self.retrieve(q)

        if lang == "english":
            system = """You are a helpful assistant.

    Answer the question ONLY using the provided context.
    If the answer is not in the context, say:
    "The answer is not present in the document."
"""
            user = f"Context:\n{context}\n\nQuestion:\n{q}"

        else:
            system = """आप एक हिन्दी सहायक हैं।

नियम:
- उत्तर केवल हिन्दी में ही दें (English बिल्कुल न लिखें)
- केवल दिए गए संदर्भ से उत्तर दें
- अगर उत्तर स्पष्ट न हो तो लिखें: "उत्तर संदर्भ में उपलब्ध नहीं है"
- कोई अनुमान न लगाएँ
"""
            user = f"""संदर्भ:
{context}

प्रश्न:
{q}

⚠️ उत्तर केवल हिन्दी में दें।
"""

        return self.generate(system, user), src


    # -----------------------------
    # SUMMARY
    # -----------------------------
    def summary(self, lang="hindi"):

        context = " ".join(self.chunks[:5])[:2000]

        if lang == "english":
            return self.generate(
                "Summarize clearly from context only",
                f"Context:\n{context}"
            )
        else:
            return self.generate(
                "केवल संदर्भ से स्पष्ट हिन्दी सारांश लिखिए",
                f"संदर्भ:\n{context}"
            )


    # -----------------------------
    # AUDIO
    # -----------------------------
    def speak(self, text, lang="hi"):

        if not text.strip():
            return

        file = f"audio_{uuid.uuid4().hex}.mp3"

        gTTS(text=text.replace("\n", " "), lang=lang).save(file)

        pygame.mixer.music.load(file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)


# =============================
# MAIN
# =============================
if __name__ == "__main__":

    MODEL = "meta-llama-3-8b-instruct.Q4_K_M.gguf"
    PDF = "data/kafan.pdf"

    rag = PDFRAG(MODEL, n_gpu_layers=-1, n_batch=2048)

    if not os.path.exists("rag_data/index.faiss"):

        print("🔧 First time setup...")

        rag.extract_text(PDF)
        rag.chunk_text()
        rag.create_index()
        rag.save()

    else:
        print("⚡ Loading saved index...")
        rag.load()

    print("\n🚀 Ready!\n")

    while True:

        q = input("Ask (or exit): ")

        if q.lower() in ["exit", "quit"]:
            break

        ans, src = rag.answer(q)

        print("\nANSWER:\n", ans)
        print("Sources:", src)