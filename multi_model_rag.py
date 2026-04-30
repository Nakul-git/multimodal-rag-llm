import json
from typing import List

from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PERSIST_DIRECTORY = "dbv1/chroma_db"
PDF_PATH = "./docs/attention-is-all-you-need.pdf"

TEXT_MODEL = "llama3.2"
IMAGE_MODEL = "llava"
EMBEDDING_MODEL = "mxbai-embed-large"

text_llm = ChatOllama(model=TEXT_MODEL, temperature=0)
image_llm = ChatOllama(model=IMAGE_MODEL, temperature=0)


def partition_document(file_path: str):
    print(f"📄 Partitioning document: {file_path}")

    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=False,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True
    )

    print(f"✅ Extracted {len(elements)} elements")
    return elements


def create_chunks_by_title(elements):
    print("🔨 Creating smart chunks...")

    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500
    )

    print(f"✅ Created {len(chunks)} chunks")
    return chunks


def separate_content_types(chunk):
    content_data = {
        "text": chunk.text,
        "tables": [],
        "images": [],
        "types": ["text"]
    }

    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            if element_type == "Table":
                content_data["types"].append("table")
                table_html = getattr(element.metadata, "text_as_html", element.text)
                content_data["tables"].append(table_html)

            elif element_type == "Image":
                if hasattr(element, "metadata") and hasattr(element.metadata, "image_base64"):
                    content_data["types"].append("image")
                    content_data["images"].append(element.metadata.image_base64)

    content_data["types"] = list(set(content_data["types"]))
    return content_data


def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    try:
        llm = image_llm if images else text_llm

        if images:
            print("     🖼️ Image found → using LLaVA")
        else:
            print("     📝 No image → using llama3.2")

        prompt_text = f"""
Create a searchable description for this PDF content.

TEXT:
{text}
"""

        if tables:
            prompt_text += "\nTABLES:\n"
            for i, table in enumerate(tables, 1):
                prompt_text += f"Table {i}:\n{table}\n\n"

        prompt_text += """
TASK:
Summarize the content for retrieval.

Include:
- key facts
- concepts
- important labels
- visible diagram information if image is provided
- questions this chunk can answer

Keep it under 400 words.
"""

        message_content = [{"type": "text", "text": prompt_text}]

        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            })

        message = HumanMessage(content=message_content)
        response = llm.invoke([message])

        return response.content

    except Exception as e:
        print(f"     ❌ AI summary failed: {e}")

        summary = text[:1000]

        if tables:
            summary += f"\n[Contains {len(tables)} table(s)]"

        if images:
            summary += f"\n[Contains {len(images)} image(s)]"

        return summary


def summarise_chunks(chunks):
    print("🧠 Processing chunks with AI summaries...")

    langchain_documents = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        print(f"   Processing chunk {current_chunk}/{total_chunks}")

        content_data = separate_content_types(chunk)

        print(f"     Types found: {content_data['types']}")
        print(f"     Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")

        if content_data["tables"] or content_data["images"]:
            print("     → Creating AI summary...")
            enhanced_content = create_ai_enhanced_summary(
                content_data["text"],
                content_data["tables"],
                content_data["images"]
            )
            print("     → AI summary created")
        else:
            enhanced_content = content_data["text"]

        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data["text"],
                    "tables_html": content_data["tables"],
                    "images_base64": content_data["images"]
                })
            }
        )

        langchain_documents.append(doc)

    print(f"✅ Processed {len(langchain_documents)} chunks")
    return langchain_documents


def create_vector_store(documents, persist_directory=PERSIST_DIRECTORY):
    print("🔮 Preparing documents for embeddings...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    safe_documents = text_splitter.split_documents(documents)

    print(f"✅ Split {len(documents)} docs into {len(safe_documents)} embedding chunks")

    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=safe_documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"✅ Vector store saved to {persist_directory}")
    return vectorstore


def run_complete_ingestion_pipeline(pdf_path: str):
    print("🚀 Starting RAG Ingestion Pipeline")
    print("=" * 50)

    elements = partition_document(pdf_path)
    chunks = create_chunks_by_title(elements)
    processed_chunks = summarise_chunks(chunks)
    db = create_vector_store(processed_chunks)

    print("🎉 Ingestion completed successfully")
    return db


if __name__ == "__main__":
    run_complete_ingestion_pipeline(PDF_PATH)