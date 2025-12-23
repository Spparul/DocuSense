from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from docx import Document
from transformers import pipeline
from rouge_score import rouge_scorer
import io

# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI(
    title="DocuSense – AI Document Analyzer",
    description="Classifies documents, summarizes content, and evaluates summary quality using ROUGE scores",
    version="1.0.0",
)

# -------------------------------------------------
# CORS
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Globals (Lazy Loaded Models)
# -------------------------------------------------
DOCUMENT_TYPES = ["Invoice", "Resume", "Legal Document", "Email"]
classifier = None
summarizer = None


def load_models():
    """Load ML models only once (important for stability)"""
    global classifier, summarizer

    if classifier is None:
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

    if summarizer is None:
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def extract_text(file: UploadFile) -> str:
    content = file.file.read()
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        return " ".join(page.extract_text() or "" for page in reader.pages)

    elif filename.endswith(".docx"):
        doc = Document(io.BytesIO(content))
        return " ".join(p.text for p in doc.paragraphs)

    elif filename.endswith(".txt"):
        return content.decode("utf-8", errors="ignore")

    return ""


def calculate_rouge(summary: str, original: str) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )
    scores = scorer.score(original[:1000], summary)

    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


# -------------------------------------------------
# API Endpoints
# -------------------------------------------------
@app.get("/")
def root():
    return {
        "project": "DocuSense",
        "message": "AI-powered document classification and summarization API",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    load_models()

    text = extract_text(file)
    if not text.strip():
        return {"error": "Could not extract text from document"}

    # ----------------------------
    # Classification
    # ----------------------------
    classification = classifier(
        text[:3000],
        DOCUMENT_TYPES
    )

    predictions = [
        {"type": label, "confidence": round(score, 4)}
        for label, score in zip(
