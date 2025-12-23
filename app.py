import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader
from docx import Document
from rouge_score import rouge_scorer
import io

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Intelligent Document Analyzer",
    layout="centered"
)

st.title("📄 Intelligent Document Classification & Summarization")

st.write(
    "Upload a document (PDF, DOCX, or TXT) to classify its type, "
    "generate a summary, and evaluate summary quality."
)

# -------------------------------
# Load Models (cached)
# -------------------------------
@st.cache_resource
def load_models():
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )
    return classifier, summarizer

classifier, summarizer = load_models()

DOCUMENT_TYPES = ["Invoice", "Resume", "Legal Document", "Email"]

# -------------------------------
# Utility functions
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

# -------------------------------
# UI: File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Document",
    type=["pdf", "docx", "txt"]
)

if uploaded_file:
    text = extract_text(uploaded_file)

    if not text.strip():
        st.error("Could not extract text from the document.")
    else:
        st.subheader("🔍 Document Classification")
        classification = classifier(text[:3000], DOCUMENT_TYPES)

        top_label = classification["labels"][0]
        confidence = classification["scores"][0]

        st.write(f"**Predicted Type:** {top_label}")
        st.write(f"**Confidence:** {confidence:.4f}")

        st.subheader("📝 Document Summary")
        summary = summarizer(
            text[:1500],
            max_length=180,
            min_length=50,
            do_sample=False
        )[0]["summary_text"]

        st.write(summary)

        st.subheader("📊 ROUGE Evaluation")
        rouge_scores = calculate_rouge(summary, text)

        for k, v in rouge_scores.items():
            st.write(f"{k}: {v}")
