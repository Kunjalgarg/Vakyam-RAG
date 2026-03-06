import os
from rag_system import PDFRAG

MODEL_PATH = "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

if not os.path.exists(MODEL_PATH):

    print("Model not found")
    exit()


rag = PDFRAG(model_path=MODEL_PATH)


pdf_path = r"D:\Language\Python\Py_projects\pdf_rag\data\kafan.pdf"

print("\nProcessing PDF...")

rag.extract_text(pdf_path)
rag.chunk_text()
rag.create_faiss_index()

AUDIO_DIR = "audio_outputs"
os.makedirs(AUDIO_DIR, exist_ok=True)

print("""
=====================================
        PDF RAG Assistant
=====================================
""")


def menu():

    print("\nChoose:")
    print("1 → Summary")
    print("2 → Q/A")
    print("3 → Both")

    choice = input("Enter (1/2/3): ").strip()

    if choice not in ["1","2","3"]:

        choice = "2"

    return choice


def language():

    l = input("\nLanguage: E (English) / H (Hindi): ").lower()

    if l == "e":
        return "english"

    return "hindi"


def output():

    print("\nOutput:")
    print("1 → Text")
    print("2 → Audio")
    print("3 → Both")

    o = input("Enter (1/2/3): ").strip()

    if o not in ["1","2","3"]:
        o = "1"

    return o

choice = menu()
lang = language()
out = output()

# ----------------
# SUMMARY
# ----------------

if choice in ["1","3"]:

    summary = rag.summarize_document(lang)

    if out in ["1","3"]:
        rag.pretty_print("SUMMARY", summary)

    if out in ["2","3"]:

        if lang == "english":
            rag.speak(summary,"en")
        else:
            rag.speak(summary,"hi")


# ----------------
# Q/A LOOP
# ----------------

while True:

    q = input("\nYour Question (type exit to quit): ")

    if q.lower() in ["exit","quit"]:
        print("\nSee you soon 😉")
        break

    ans, src = rag.answer_question(q, lang)

    rag.pretty_print("ANSWER", ans)

    print("Sources:", ", ".join(src))

    if out in ["2","3"]:

        if lang == "english":
            rag.speak(ans,"en")
        else:
            rag.speak(ans,"hi")