import json
import time
import hashlib
import concurrent.futures
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from sentence_transformers import CrossEncoder

load_dotenv()

PERSIST_DIRECTORY = "dbv1/chroma_db"

TEXT_MODEL = "llama3.2"
IMAGE_MODEL = "llava"
EMBEDDING_MODEL = "mxbai-embed-large"

RRF_CANDIDATE_CHUNKS = 15
RETRIEVED_CHUNKS = 5
MULTI_QUERY_COUNT = 3
RRF_K = 60

MAX_IMAGES_TO_ANALYZE = 3
IMAGE_TIMEOUT_SECONDS = 120
LIVE_UPDATE_EVERY_SECONDS = 5

USE_MMR = True
USE_HYBRID_SEARCH = True
USE_RERANKER = True

BM25_K = 5
DEBUG_EXPORT_JSON = True

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

text_llm = ChatOllama(model=TEXT_MODEL, temperature=0)
image_llm = ChatOllama(model=IMAGE_MODEL, temperature=0)

print("🔁 Loading MiniLM reranker...", flush=True)
reranker_model = CrossEncoder(RERANKER_MODEL)
print("✅ MiniLM reranker loaded", flush=True)


class QueryVariations(BaseModel):
    queries: List[str]


def load_vector_store():
    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)

    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    return db


def load_documents_from_chroma(db):
    print("📦 Loading Chroma docs for BM25 keyword search...", flush=True)

    data = db.get(include=["documents", "metadatas"])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    bm25_docs = []

    for i, text in enumerate(documents):
        if not text:
            continue

        metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}

        bm25_docs.append(
            Document(
                page_content=text,
                metadata=metadata
            )
        )

    print(f"✅ Loaded {len(bm25_docs)} docs for BM25", flush=True)
    return bm25_docs


def create_vector_retriever(db, k=RRF_CANDIDATE_CHUNKS):
    if USE_MMR:
        return db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": 25,
                "lambda_mult": 0.7
            }
        )

    return db.as_retriever(search_kwargs={"k": k})


def create_bm25_retriever(db):
    bm25_documents = load_documents_from_chroma(db)

    if not bm25_documents:
        return None

    bm25_retriever = BM25Retriever.from_documents(bm25_documents)
    bm25_retriever.k = BM25_K

    print("✅ BM25 keyword retriever ready", flush=True)
    return bm25_retriever


def generate_query_variations(original_query):
    print("\n🧠 Generating multi-query variations...", flush=True)

    structured_llm = text_llm.with_structured_output(QueryVariations)

    prompt = f"""
Generate {MULTI_QUERY_COUNT} different search query variations for this question.

Original query:
{original_query}

Rules:
- Keep the meaning same.
- Use different wording.
- Make them useful for document retrieval.
- Return only structured output.
"""

    try:
        response = structured_llm.invoke(prompt)
        queries = response.queries
    except Exception as e:
        print(f"⚠️ Query variation failed: {e}", flush=True)
        return [original_query]

    clean_queries = []

    for q in queries:
        q = q.strip()
        if q and q not in clean_queries:
            clean_queries.append(q)

    if original_query not in clean_queries:
        clean_queries.insert(0, original_query)

    print("✅ Query variations created:", flush=True)
    for i, q in enumerate(clean_queries, 1):
        print(f"   {i}. {q}", flush=True)

    return clean_queries


def get_doc_id(doc):
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page", "")
    content = doc.page_content[:500]

    unique_string = f"{source}|{page}|{content}"
    return hashlib.md5(unique_string.encode("utf-8")).hexdigest()


def reciprocal_rank_fusion(all_results, final_k=RRF_CANDIDATE_CHUNKS, rrf_k=RRF_K):
    print("\n🔀 Applying RRF fusion...", flush=True)
    print(f"🔀 RRF received {len(all_results)} result lists", flush=True)

    scores = {}
    doc_map = {}

    for result_list in all_results:
        for rank, doc in enumerate(result_list, start=1):
            doc_id = get_doc_id(doc)

            if doc_id not in scores:
                scores[doc_id] = 0
                doc_map[doc_id] = doc

            scores[doc_id] += 1 / (rrf_k + rank)

    ranked_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    fused_docs = [doc_map[doc_id] for doc_id in ranked_doc_ids[:final_k]]

    print(f"✅ RRF selected top {len(fused_docs)} candidate chunks", flush=True)
    return fused_docs


def rerank_chunks(query, chunks, top_n=RETRIEVED_CHUNKS):
    if not USE_RERANKER:
        print("⏭️ Reranker disabled", flush=True)
        return chunks[:top_n]

    if not chunks:
        return []

    print("\n🎯 Applying MiniLM reranker...", flush=True)

    pairs = [(query, doc.page_content) for doc in chunks]

    try:
        scores = reranker_model.predict(pairs)

        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        reranked_docs = [doc for doc, score in ranked[:top_n]]

        print(f"✅ MiniLM reranked {len(chunks)} chunks → top {len(reranked_docs)} chunks", flush=True)

        for i, (doc, score) in enumerate(ranked[:top_n], 1):
            preview = doc.page_content[:120].replace("\n", " ")
            print(f"   Rank {i} | Score: {score:.4f} | {preview}...", flush=True)

        return reranked_docs

    except Exception as e:
        print(f"⚠️ MiniLM reranker failed: {e}", flush=True)
        print("⚠️ Falling back to RRF chunks", flush=True)
        return chunks[:top_n]


def retrieve_chunks(db, query, k=RETRIEVED_CHUNKS):
    query_variations = generate_query_variations(query)

    vector_retriever = create_vector_retriever(db, k=RRF_CANDIDATE_CHUNKS)

    bm25_retriever = None
    if USE_HYBRID_SEARCH:
        print("\n🔎 Using HYBRID SEARCH: Vector + BM25 keyword", flush=True)
        bm25_retriever = create_bm25_retriever(db)
    else:
        print("\n🔎 Using VECTOR SEARCH only", flush=True)

    all_retrieval_results = []

    for i, variation in enumerate(query_variations, 1):
        print(f"\n🔍 Searching with Query {i}: {variation}", flush=True)

        vector_docs = vector_retriever.invoke(variation)
        print(f"✅ Vector/MMR retrieved {len(vector_docs)} chunks", flush=True)

        all_retrieval_results.append(vector_docs)

        if USE_HYBRID_SEARCH and bm25_retriever is not None:
            bm25_docs = bm25_retriever.invoke(variation)
            print(f"✅ BM25 retrieved {len(bm25_docs)} chunks", flush=True)
            all_retrieval_results.append(bm25_docs)

        for j, doc in enumerate(vector_docs[:3], 1):
            preview = doc.page_content[:150].replace("\n", " ")
            print(f"   Preview Doc {j}: {preview}...", flush=True)

    rrf_chunks = reciprocal_rank_fusion(
        all_results=all_retrieval_results,
        final_k=RRF_CANDIDATE_CHUNKS
    )

    final_chunks = rerank_chunks(
        query=query,
        chunks=rrf_chunks,
        top_n=k
    )

    print(f"\n✅ Final chunks after Hybrid + Multi-Query + RRF + MiniLM: {len(final_chunks)}", flush=True)

    return final_chunks


def export_chunks_to_json(chunks, output_file="rag_results.json"):
    results = []

    for i, chunk in enumerate(chunks, 1):
        results.append({
            "chunk_number": i,
            "page_content": chunk.page_content,
            "metadata": chunk.metadata
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"✅ Retrieved chunks exported to {output_file}", flush=True)


def extract_original_content(chunk):
    raw = chunk.metadata.get("original_content")

    if not raw:
        return {}

    try:
        return json.loads(raw)
    except Exception:
        return {}


def build_text_context_from_chunks(chunks):
    context = ""

    for i, chunk in enumerate(chunks, 1):
        context += f"\n--- Document {i} ---\n"

        original_data = extract_original_content(chunk)

        if original_data:
            raw_text = original_data.get("raw_text", "")
            tables_html = original_data.get("tables_html", [])

            if raw_text:
                context += f"TEXT:\n{raw_text}\n\n"

            if tables_html:
                context += "TABLES:\n"
                for j, table in enumerate(tables_html, 1):
                    context += f"Table {j}:\n{table}\n\n"
        else:
            context += chunk.page_content + "\n"

    return context


def collect_images_one_by_one(chunks, max_images=MAX_IMAGES_TO_ANALYZE):
    images = []

    for chunk_index, chunk in enumerate(chunks, 1):
        original_data = extract_original_content(chunk)
        images_base64 = original_data.get("images_base64", [])

        for image_index, image_base64 in enumerate(images_base64, 1):
            if not image_base64:
                continue

            images.append({
                "chunk_number": chunk_index,
                "image_number": image_index,
                "image_base64": image_base64
            })

            if len(images) >= max_images:
                return images

    return images


def analyze_single_image(image_base64, query, chunk_number, image_number):
    prompt_text = f"""
Question:
{query}

You are analyzing one image from a retrieved PDF chunk.

Image location:
Chunk {chunk_number}, Image {image_number}

Extract only useful visual information for answering the question.

Focus on:
- labels
- arrows
- diagram blocks
- architecture parts
- tables/charts
- important visible text

Keep it short.
Do not guess.
"""

    message_content = [
        {"type": "text", "text": prompt_text},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        }
    ]

    message = HumanMessage(content=message_content)
    response = image_llm.invoke([message])
    return response.content


def analyze_single_image_with_live_updates(image_item, query, current_index, total_images):
    chunk_number = image_item["chunk_number"]
    image_number = image_item["image_number"]
    image_base64 = image_item["image_base64"]

    print("\n" + "-" * 60, flush=True)
    print(f"📸 Preparing Image {current_index}/{total_images}", flush=True)
    print(f"📍 Source: Chunk {chunk_number}, Image {image_number}", flush=True)
    print(f"📤 LLaVA received Image {current_index}", flush=True)
    print(f"🧠 LLaVA started processing Image {current_index}...", flush=True)

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            analyze_single_image,
            image_base64,
            query,
            chunk_number,
            image_number
        )

        while True:
            if future.done():
                try:
                    summary = future.result()
                    elapsed = round(time.time() - start_time, 2)

                    if summary:
                        print(
                            f"✅ LLaVA successfully processed Image {current_index}/{total_images} "
                            f"in {elapsed} seconds",
                            flush=True
                        )
                        return summary

                    print(f"⚠️ LLaVA returned empty result for Image {current_index}", flush=True)
                    return ""

                except Exception as e:
                    elapsed = round(time.time() - start_time, 2)
                    print(f"❌ LLaVA failed on Image {current_index} after {elapsed}s", flush=True)
                    print(f"Error: {e}", flush=True)
                    return ""

            elapsed = round(time.time() - start_time, 2)

            if elapsed >= IMAGE_TIMEOUT_SECONDS:
                print(f"⏱️ Timeout: Image {current_index} took more than {IMAGE_TIMEOUT_SECONDS}s", flush=True)
                return ""

            print(
                f"⏳ LLaVA still processing Image {current_index}/{total_images} "
                f"| {elapsed}s elapsed...",
                flush=True
            )

            time.sleep(LIVE_UPDATE_EVERY_SECONDS)


def analyze_images_one_by_one(chunks, query):
    images = collect_images_one_by_one(chunks)

    if not images:
        print("📝 No images found in retrieved chunks", flush=True)
        return []

    total_images = len(images)

    print(f"🖼️ Found {total_images} relevant image(s)", flush=True)
    print("🐢 Processing images one-by-one with LLaVA", flush=True)

    image_summaries = []

    for index, image_item in enumerate(images, 1):
        print(f"\n➡️ Starting Image {index}/{total_images}", flush=True)

        summary = analyze_single_image_with_live_updates(
            image_item=image_item,
            query=query,
            current_index=index,
            total_images=total_images
        )

        if summary:
            image_summaries.append({
                "chunk_number": image_item["chunk_number"],
                "image_number": image_item["image_number"],
                "summary": summary
            })

        if index < total_images:
            print(f"➡️ Moving to Image {index + 1}/{total_images}", flush=True)
        else:
            print("🏁 Finished all image processing", flush=True)

    print("\n" + "-" * 60, flush=True)
    print(f"✅ Created {len(image_summaries)} image summary/summaries", flush=True)
    return image_summaries


def build_image_context(image_summaries):
    if not image_summaries:
        return "No image analysis available."

    image_context = ""

    for item in image_summaries:
        image_context += f"""
--- Image Summary ---
Chunk: {item["chunk_number"]}
Image: {item["image_number"]}
Description:
{item["summary"]}
"""

    return image_context


def generate_final_answer(chunks, query):
    try:
        text_context = build_text_context_from_chunks(chunks)
        image_summaries = analyze_images_one_by_one(chunks, query)
        image_context = build_image_context(image_summaries)

        print("\n🧠 Generating final answer using llama3.2...", flush=True)

        final_prompt = f"""
You are a helpful RAG chatbot.

Answer the user's question using ONLY the retrieved PDF content.

QUESTION:
{query}

RETRIEVED TEXT AND TABLE CONTENT:
{text_context}

RETRIEVED IMAGE ANALYSIS BY LLAVA:
{image_context}

RULES:
1. Answer clearly.
2. Use text, tables, and LLaVA image analysis if useful.
3. Do not make up facts.
4. If the answer is not present, say:
"I don't have enough information to answer that question based on the provided documents."

ANSWER:
"""

        message = HumanMessage(content=final_prompt)
        response = text_llm.invoke([message])

        return response.content

    except Exception as e:
        print(f"❌ Answer generation failed: {e}", flush=True)
        return "Sorry, I encountered an error while generating the answer."


def run_single_query(db, query):
    chunks = retrieve_chunks(db, query, k=RETRIEVED_CHUNKS)

    if DEBUG_EXPORT_JSON:
        export_chunks_to_json(chunks, "rag_results.json")

    answer = generate_final_answer(chunks, query)
    return answer


def run_chatbot():
    print("🤖 Starting Hybrid + Multi-Query + RRF + MiniLM + LLaVA RAG Chatbot", flush=True)
    print("=" * 85, flush=True)
    print("Type your question and press Enter.", flush=True)
    print("Type 'exit', 'quit', or 'q' to stop.", flush=True)
    print("=" * 85, flush=True)

    db = load_vector_store()

    while True:
        query = input("\nYou: ").strip()

        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Chatbot stopped.", flush=True)
            break

        if not query:
            print("Please ask a question.", flush=True)
            continue

        print("\n🔍 Searching documents with Hybrid + Multi-Query + RRF + MiniLM...", flush=True)
        answer = run_single_query(db, query)

        print("\n🤖 Answer:", flush=True)
        print(answer, flush=True)


if __name__ == "__main__":
    run_chatbot()