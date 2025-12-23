from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from docx import Document
from transformers import pipeline
from rouge_score import rouge_scorer
import io
import re

app = FastAPI()

# Allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AI models (downloads on first run)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

DOCUMENT_TYPES = ["Invoice", "Resume", "Legal Document", "Email"]

def extract_text(file: UploadFile) -> str:
    """Extract text from PDF, DOCX, or TXT files"""
    content = file.file.read()
    filename = file.filename.lower()
    
    if filename.endswith('.pdf'):
        reader = PdfReader(io.BytesIO(content))
        return " ".join(page.extract_text() or "" for page in reader.pages)
    elif filename.endswith('.docx'):
        doc = Document(io.BytesIO(content))
        return " ".join(para.text for para in doc.paragraphs)
    elif filename.endswith('.txt') or filename.endswith('.eml'):
        return content.decode('utf-8', errors='ignore')
    return ""

def calculate_rouge(summary: str, original: str) -> dict:
    """Calculate ROUGE scores"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(original[:1000], summary)
    return {
        "rouge1": round(scores['rouge1'].fmeasure, 4),
        "rouge2": round(scores['rouge2'].fmeasure, 4),
        "rougeL": round(scores['rougeL'].fmeasure, 4)
    }

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    # Extract text
    text = extract_text(file)
    if not text.strip():
        return {"error": "Could not extract text from document"}
    
    # Truncate for models (they have token limits)
    text_for_analysis = text[:4000]
    
    # Classify document type
    classification = classifier(text_for_analysis, DOCUMENT_TYPES, multi_label=True)
    predictions = [
        {"type": label, "confidence": round(score, 4)}
        for label, score in zip(classification['labels'], classification['scores'])
    ]
    
    # Generate summary (BART has 1024 token limit)
    text_for_summary = text[:2000]
    summary_result = summarizer(
        text_for_summary, 
        max_length=200, 
        min_length=50, 
        do_sample=False
    )
    summary = summary_result[0]['summary_text']
    
    # Calculate ROUGE scores
    rouge_scores = calculate_rouge(summary, text)
    
    return {
        "filename": file.filename,
        "textLength": len(text),
        "predictions": predictions,
        "topPrediction": predictions[0] if predictions else None,
        "accuracy": predictions[0]['confidence'] if predictions else 0,
        "summary": summary,
        "rougeScores": rouge_scores
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}
