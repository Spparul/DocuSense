import streamlit as st
import re
from PyPDF2 import PdfReader
from docx import Document
from rouge_score import rouge_scorer
from transformers import pipeline

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="AIML Project 2", layout="centered")
st.title("📄 Intelligent Document Classification & Summarization")

st.write(
    "Upload a document to automatically identify its type, "
    "generate a summary, and evaluate summary quality."
)

# -------------------------------
# Load summarizer (cached)
# -------------------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# -------------------------------
# Utilities
# -------------------------------
def extract_text(file):
    name = file.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(file)
        return " ".join(p.extract_text() or "" for p in reader.pages)

    elif name.endswith(".docx"):
        doc = Document(file)
        return " ".join(p.text for p in doc.paragraphs)

    elif name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    return ""

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -------------------------------
# DOCUMENT CLASSIFICATION (STABLE)
# -------------------------------
def classify_document(text):
    t = text.lower()

    if any(k in t for k in ["invoice", "gst", "total amount", "bill to"]):
        return "Invoice", 0.95

    if any(k in t for k in ["resume", "skills", "experience", "education"]):
        return "Resume", 0.95

    if any(k in t for k in ["agreement", "hereby", "party", "clause"]):
        return "Legal Document", 0.95

    if any(k in t for k in ["dear", "regards", "from:", "to:"]):
        return "Email", 0.95

    return "Unknown / Mixed", 0.50

def concise_summary(text, max_sentences=3):
    sentences = re.split(r'(?<=[.!?]) +', text)

    # Remove very short / useless sentences
    sentences = [s for s in sentences if len(s.split()) > 6]

    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    # Score sentences by keyword importance
    keywords = [
        "experience", "skills", "education", "invoice", "amount",
        "agreement", "payment", "project", "responsibilities"
    ]

    scored = []
    for s in sentences:
        score = sum(1 for k in keywords if k in s.lower())
        scored.append((score, s))

    # Sort by score and keep top sentences
    scored.sort(reverse=True, key=lambda x: x[0])
    top_sentences = [s for _, s in scored[:max_sentences]]

    return " ".join(top_sentences)


# -------------------------------
# ROUGE
# -------------------------------
def calculate_rouge(summary, original):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(original[:1000], summary)
    return {
        "ROUGE-1": round(scores["rouge1"].fmeasure, 4),
        "ROUGE-2": round(scores["rouge2"].fmeasure, 4),
        "ROUGE-L": round(scores["rougeL"].fmeasure, 4),
    }

# -------------------------------
# UI FLOW
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"]
)

if uploaded_file:
    text = extract_text(uploaded_file)
    text = clean_text(text)

    if not text:
        st.error("Could not extract text.")
    else:
        doc_type, confidence = classify_document(text)

        st.subheader("📌 Document Classification")
        st.write(f"**Type:** {doc_type}")
        st.write(f"**Confidence:** {confidence}")

        st.subheader("📝 Document Summary")
        summary = concise_summary(text)
        st.write(summary)

        st.subheader("📊 ROUGE Scores")
        rouge = calculate_rouge(summary, text)
        for k, v in rouge.items():
            st.write(f"{k}: {v}")
