import torch
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

from src.ocr import extract_text_from_pdf
from src.retriever import Retriever
from src.audio import Audio


class PDFRAG:
    def __init__(self, model_path=None):
        self.model_path = model_path

        # components
        from src.loader import DocumentLoader
        from src.chunker import TextChunker
        from src.embedder import Embedder
        from src.retriever import Retriever

        self.loader = DocumentLoader("data/raw")
        self.chunker = TextChunker()
        self.embedder = Embedder()
        self.retriever = Retriever()

        # ✅ ONLY load LLM if model_path is provided
        self.llm = None
        if model_path is not None:
            from llama_cpp import Llama
            print("Loading LLM...")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_threads=8
            )

    # -------------------------
    def process_pdf(self, pdf_path):

        text = extract_text_from_pdf(pdf_path)
        self.chunks = self.chunk_text(text)

        self.retriever.build(self.chunks)
        self.retriever.save()

    # -------------------------
    def chunk_text(self, text, size=300):

        sentences = text.split("।")

        chunks = []
        current = ""

        for s in sentences:
            if len(current) + len(s) < size:
                current += s + "।"
            else:
                chunks.append(current)
                current = s + "।"

        if current:
            chunks.append(current)

        return chunks

    # -------------------------
    def load(self):
        self.retriever.load()

    # -------------------------
    def generate(self, context, question):

        prompt = f"""
Answer ONLY from context.

Context:
{context}

Question:
{question}
"""

        out = self.llm(prompt, max_tokens=300)

        return out["choices"][0]["text"].strip()

    # -------------------------
    def answer(self, question):

        context, src = self.retriever.search(question)
        ans = self.generate(context, question)

        return ans, src