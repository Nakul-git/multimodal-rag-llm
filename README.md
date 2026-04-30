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

