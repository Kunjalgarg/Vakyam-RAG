import os
import faiss
import numpy as np
import difflib
import uuid
import psutil
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from pdf2image import convert_from_path
import pytesseract
from gtts import gTTS
import pygame


class PDFRAG:

    def __init__(self, model_path):

        print("Loading LLM...")

        physical = psutil.cpu_count(logical=False)
        threads = max(1, physical - 1)

        print(f"System Detected: {physical} Physical Cores")
        print(f"Setting n_threads to: {threads}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=threads,
            n_batch=512,
            verbose=False
        )

        print("Loading embedding model...")

        self.embedding_model = SentenceTransformer(
            "intfloat/multilingual-e5-base",
            device="cpu"
        )

        self.chunks = []
        self.index = None


    # -----------------------------
    # TEXT EXTRACTION
    # -----------------------------

    def extract_text(self, pdf_path):

        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        poppler_path = r"C:\poppler\poppler-25.12.0\Library\bin"

        pages = convert_from_path(pdf_path, dpi=200, poppler_path=poppler_path)

        text = ""

        for page in pages:
            t = pytesseract.image_to_string(page, lang="hin")
            text += t + "\n"

        self.full_text = text
        return text


    # -----------------------------
    # CHUNKING
    # -----------------------------

    def chunk_text(self, chunk_size=800, overlap=120):

        text = self.full_text

        self.chunks = [
            text[i:i + chunk_size]
            for i in range(0, len(text), chunk_size - overlap)
        ]

        return self.chunks


    # -----------------------------
    # FAISS INDEX
    # -----------------------------

    def create_faiss_index(self):

        passages = ["passage: " + c for c in self.chunks]

        emb = self.embedding_model.encode(passages, convert_to_numpy=True)

        dim = emb.shape[1]

        self.index = faiss.IndexFlatIP(dim)

        faiss.normalize_L2(emb)

        self.index.add(emb)


    # -----------------------------
    # QUERY UNDERSTANDING
    # -----------------------------

    def normalize_query(self, query):

        query = query.lower()

        words = query.split()

        dictionary = [
            "kafan",
            "ghisu",
            "madhav",
            "budiya",
            "summary",
            "story",
            "death",
            "poverty"
        ]

        fixed = []

        for w in words:

            match = difflib.get_close_matches(w, dictionary, n=1)

            if match:
                fixed.append(match[0])
            else:
                fixed.append(w)

        return " ".join(fixed)


    # -----------------------------
    # RETRIEVAL
    # -----------------------------

    def retrieve(self, query, k=3):

        query = self.normalize_query(query)

        q = self.embedding_model.encode(
            ["query: " + query],
            convert_to_numpy=True
        )

        faiss.normalize_L2(q)

        scores, idx = self.index.search(q, k)

        context = []

        sources = []

        for i in idx[0]:

            context.append(self.chunks[i])

            sources.append(f"Chunk {i}")

        return "\n".join(context), sources


    # -----------------------------
    # GENERATION
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

        output = self.llm(
            prompt,
            max_tokens=600,
            temperature=0.1,
            repeat_penalty=1.2,
            stop=["<|eot_id|>"]
        )

        return output["choices"][0]["text"].strip()


    # -----------------------------
    # SUMMARY
    # -----------------------------

    def summarize_document(self, language="hindi"):

        context = " ".join(self.chunks[:4])
        context = context[:2000]   # prevent context overflow

        if language == "english":

            system = """
    You are a literature expert.

    Write a clear and detailed summary of the story using ONLY the provided context.
    Do not introduce yourself.
    Do not explain instructions.
    Do not add outside knowledge.
    Write in English.
    """

            user = f"""
    Context:
    {context}

    Write a detailed summary of the story.
    """

        else:

            system = """
    आप एक साहित्य विशेषज्ञ हैं।

    नीचे दिए गए पाठ के आधार पर कहानी का स्पष्ट और विस्तृत सारांश लिखिए।
    • केवल हिन्दी में लिखें
    • अपना परिचय न दें
    • निर्देशों की व्याख्या न करें
    • बाहरी जानकारी न जोड़ें
    """

            user = f"""
    संदर्भ:
    {context}

    कहानी का विस्तृत हिन्दी सारांश लिखिए।
    """

        return self.generate(system, user)


    # -----------------------------
    # QUESTION ANSWER
    # -----------------------------
    def answer_question(self, question, language="hindi"):

        context, sources = self.retrieve(question)

        if language == "english":

            system = """
    You are a helpful assistant.

    Answer the question ONLY using the provided context.
    If the answer is not in the context, say:
    "The answer is not present in the document."
    """

            user = f"""
    Context:
    {context}

    Question:
    {question}

    Answer clearly in English.
    """

        else:

            system = """
    आप एक सहायक AI हैं।

    केवल दिए गए संदर्भ के आधार पर उत्तर दें।
    यदि उत्तर संदर्भ में नहीं है तो लिखें:
    "उत्तर दस्तावेज़ में उपलब्ध नहीं है।"
    """

            user = f"""
    संदर्भ:
    {context}

    प्रश्न:
    {question}

    स्पष्ट हिन्दी में उत्तर दें।
    """

        answer = self.generate(system, user)

        return answer, sources

    # -----------------------------
    # AUDIO
    # -----------------------------

    def speak(self, text, lang="en"):

        if not text or not text.strip():
            print("No text for audio.")
            return

        AUDIO_DIR = "audio_outputs"
        os.makedirs(AUDIO_DIR, exist_ok=True)

        filename = os.path.join(AUDIO_DIR, f"audio_{uuid.uuid4().hex}.mp3")

        tts = gTTS(text=text.replace("\n"," "), lang=lang)
        tts.save(filename)

        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            continue

        pygame.mixer.quit()

        print(f"\nAudio saved → {filename}")

    # -----------------------------
    # PRETTY PRINT
    # -----------------------------

    def pretty_print(self, title, text):

        print("\n" + "=" * 60)
        print(f"{title}")
        print("=" * 60)

        print(text)

        print("=" * 60)