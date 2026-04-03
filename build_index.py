import sys
import os

sys.path.append(os.path.abspath("."))
from src.rag_system import PDFRAG

rag = PDFRAG()   # no model needed

print("🔨 Building index...")
rag.build()

print("✅ Index built successfully!")