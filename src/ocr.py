import pytesseract
from pdf2image import convert_from_path


def extract_text_from_pdf(pdf_path):
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    poppler_path = r"C:\\poppler\\poppler-25.12.0\\Library\\bin"

    pages = convert_from_path(pdf_path, dpi=200, poppler_path=poppler_path)

    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page, lang="hin") + "\n"

    return clean_text(text)


def clean_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())

    text = text.replace("म ा ध व", "माधव")
    text = text.replace("घ ी स ू", "घीसू")

    return text