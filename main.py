import os
from rag_system import PDFRAG

# ✅ Force GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_PATH = "meta-llama-3-8b-instruct.Q4_K_M.gguf"
DATA_PATH = "rag_data"

if not os.path.exists(MODEL_PATH):
    print("Model not found")
    exit()

rag = PDFRAG(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_batch=2048
)

pdf_path = r"data\kafan.pdf"

print("\nInitializing RAG...")

# =============================
# BUILD OR LOAD
# =============================
if not os.path.exists(os.path.join(DATA_PATH, "index.faiss")):

    print("\n🔧 First-time setup (building index)...")

    rag.extract_text(pdf_path)
    rag.chunk_text()
    rag.create_index()        # ✅ FIXED
    rag.save(DATA_PATH)       # ✅ FIXED

else:
    print("\n⚡ Loading existing index...")
    rag.load(DATA_PATH)       # ✅ FIXED


# =============================
# AUDIO DIR
# =============================
AUDIO_DIR = "audio_outputs"
os.makedirs(AUDIO_DIR, exist_ok=True)

print("""
=====================================
        PDF RAG Assistant
=====================================
""")

# ----------------
# MENU
# ----------------
def menu():
    print("\nChoose:")
    print("1 → Summary")
    print("2 → Q/A")
    print("3 → Both")

    choice = input("Enter (1/2/3): ").strip()
    return choice if choice in ["1", "2", "3"] else "2"


def language():
    l = input("\nLanguage: E (English) / H (Hindi): ").lower()
    return "english" if l == "e" else "hindi"


def output_mode():
    print("\nOutput:")
    print("1 → Text")
    print("2 → Audio")
    print("3 → Both")

    o = input("Enter (1/2/3): ").strip()
    return o if o in ["1", "2", "3"] else "1"


choice = menu()
lang = language()
out = output_mode()

# ----------------
# SUMMARY
# ----------------
if choice in ["1", "3"]:
    print("\n⚡ Generating summary...")

    summary = rag.summary(lang)   # ✅ FIXED

    if out in ["1", "3"]:
        print("\n===== SUMMARY =====\n")
        print(summary)

    if out in ["2", "3"]:
        rag.speak(summary, "en" if lang == "english" else "hi")


# ----------------
# Q/A LOOP
# ----------------
while True:
    q = input("\nYour Question (type exit to quit): ")

    if q.lower() in ["exit", "quit"]:
        print("\nSee you soon 😉")
        break

    print("\n⚡ Thinking (GPU)...")

    ans, src = rag.answer(q, lang)   # ✅ FIXED

    print("\n===== ANSWER =====\n")
    print(ans)
    print("\nSources:", ", ".join(src))

    if out in ["2", "3"]:
        rag.speak(ans, "en" if lang == "english" else "hi")