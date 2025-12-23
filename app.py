import streamlit as st
import re
from PyPDF2 import PdfReader
from docx import Document
from rouge_score import rouge_scorer

# -------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="AIML Project 2 – Intelligent Document Analyzer",
    layout="centered"
)

st.title("📄 Intelligent Document Classification & Summarization")
st.write(
    "Upload a document to automatically identify its type, "
    "generate a concise summary, and evaluate summary quality."
)

# -------------------------------------------------
# Text Extraction
# -------------------------------------------------
def extract_text(file):
    name = file.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(file)
        return " ".join(page.extract_text() or "" for page in reader.pages)

    elif name.endswith(".docx"):
        doc = Document(file)
        return " ".join(p.text for p in doc.paragraphs)

    elif name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    return ""

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -------------------------------------------------
# DOCUMENT CLASSIFICATION (RULE-BASED, STABLE)
# -------------------------------------------------
def classify_document(text):
    t = text.lower()

    if any(k in t for k in ["invoice", "gst", "bill to", "total amount", "tax"]):
        return "Invoice", 0.95

    if any(k in t for k in ["resume", "curriculum vitae", "education", "skills", "experience"]):
        return "Resume", 0.95

    if any(k in t for k in ["agreement", "hereby", "party of the", "clause", "act,"]):
        return "Legal Document", 0.95

    if any(k in t for k in ["dear", "regards", "from:", "to:", "subject:"]):
        return "Email", 0.95

    return "General / Mixed Document", 0.60

# -------------------------------------------------
# CONCISE, SECTION-AWARE SUMMARY
# -------------------------------------------------
def generate_summary(text, doc_type):
    lines = [l.strip() for l in text.split(".") if len(l.strip()) > 40]
    summary_lines = []

    for line in lines:
        l = line.lower()

        if doc_type == "Resume":
            if any(k in l for k in ["experience", "skills", "education", "project"]):
                summary_lines.append(line)

        elif doc_type == "Invoice":
            if any(k in l for k in ["total", "amount", "gst", "invoice"]):
                summary_lines.append(line)

        elif doc_type == "Legal Document":
            if any(k in l for k in ["agreement", "obligation", "clause", "terms"]):
                summary_lines.append(line)

        elif doc_type == "Email":
            if any(k in l for k in ["request", "thank", "regards", "inform"]):
                summary_lines.append(line)

    summary_lines = summary_lines[:3]

    if summary_lines:
        return ". ".join(summary_lines) + "."
    else:
        return "This document contains important information relevant to the identified category."

# -------------------------------------------------
# ROUGE SCORE
# -------------------------------------------------
def calculate_rouge(summary, original):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )
    scores = scorer.score(original[:1000], summary)
    return {
        "ROUGE-1": round(scores["rouge1"].fmeasure, 4),
        "ROUGE-2": round(scores["rouge2"].fmeasure, 4),
        "ROUGE-L": round(scores["rougeL"].fmeasure, 4),
    }

# -------------------------------------------------
# UI FLOW
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload PDF / DOCX / TXT",
    type=["pdf", "docx", "txt"]
)

if uploaded_file:
    raw_text = extract_text(uploaded_file)
    text = clean_text(raw_text)

    if not text:
        st.error("Could not extract readable text from the document.")
    else:
        doc_type, confidence = classify_document(text)

        st.subheader("📌 Document Classification")
        st.write(f"**Type:** {doc_type}")
        st.write(f"**Confidence:** {confidence}")

        st.subheader("📝 Concise Summary")
        summary = generate_summary(text, doc_type)
        st.write(summary)

        st.subheader("📊 ROUGE Evaluation")
        rouge_scores = calculate_rouge(summary, text)
        for k, v in rouge_scores.items():
            st.write(f"{k}: {v}")
