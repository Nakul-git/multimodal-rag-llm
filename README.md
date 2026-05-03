# 🧠 Multimodal RAG System (Local, Hybrid, Vision-Enhanced)

A **production-style Multimodal RAG (Retrieval-Augmented Generation) system** that can ingest PDFs with **text, tables, and images**, and answer questions using advanced retrieval and reasoning techniques.

---

## 🚀 Key Features

- 📄 Smart PDF parsing using Unstructured  
- 🧠 AI-enhanced chunk summaries (text + image aware)  
- 🔍 Hybrid search (semantic + keyword)  
- 🔁 Multi-query retrieval for better recall  
- 🧮 Reciprocal Rank Fusion (RRF)  
- 🎯 MiniLM reranker for precision  
- 🖼️ LLaVA for visual reasoning  
- 🧾 Structured JSON debug output  
- 💻 Fully local (no API required)  

---

## 🏗️ Architecture

```text
User Query
   ↓
Multi-Query Expansion
   ↓
Hybrid Retrieval
   → Vector Search (MMR)
   → BM25 Keyword Search
   ↓
RRF (Reciprocal Rank Fusion)
   ↓
MiniLM Reranker
   ↓
Final Top Chunks
   ↓
LLaVA Image Analysis
   ↓
llama3.2 Final Answer
```
---

## ⚙️ Tech Stack

| Component    | Technology            |
|--------------|----------------------|
| LLM          | Ollama (llama3.2)    |
| Vision Model | LLaVA                |
| Embeddings   | mxbai-embed-large    |
| Vector DB    | ChromaDB             |
| Reranker     | MiniLM (CrossEncoder)|
| Retrieval    | BM25 + Vector + MMR  |
| PDF Parsing  | Unstructured         |
| Framework    | LangChain            |

---

## 📦 Installation

```bash
git clone https://github.com/your-username/multimodal-rag.git
cd multimodal-rag

python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

---

## 🤖 Download Models (Ollama)

```bash
ollama pull llama3.2
ollama pull llava
ollama pull mxbai-embed-large
```

## 📄 Dataset / Knowledge Source

This project uses the Transformer research paper:

📘 **Attention Is All You Need**  
📁 Located in: `docs/attention-is-all-you-need.pdf`

---

## 🧩 System Dependencies (Important for Windows Users)

To ensure smooth execution of the Multimodal RAG pipeline (PDF parsing, OCR, and file handling), you need to install the following system-level dependencies:

🪟 Windows Setup

##1️⃣ Install Poppler (for PDF processing)

Poppler is required for extracting text/images from PDFs.

Download: https://github.com/oschwartz10612/poppler-windows/releases
Extract the ZIP file
Add the bin folder to your system PATH

Example:

C:\poppler-xx\Library\bin

##2️⃣ Install Tesseract OCR (for image text extraction)

Tesseract is used for OCR (reading text from images inside PDFs).

Download: https://github.com/UB-Mannheim/tesseract/wiki
Install it (default path recommended)

Example path:

C:\Program Files\Tesseract-OCR\tesseract.exe

👉 Add to PATH or set manually in code:

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

##3️⃣ Install libmagic (file type detection)

Required for handling file formats properly.

Install via pip (Windows-compatible):

pip install python-magic-bin

---

🍎 macOS Setup

brew install poppler
brew install tesseract
brew install libmagic

---

🐧 Linux Setup (Ubuntu/Debian)

sudo apt update
sudo apt install -y poppler-utils tesseract-ocr libmagic1

---

✅ Verify Installation

Run these commands to confirm everything is working:

tesseract --version
pdftoppm -h



