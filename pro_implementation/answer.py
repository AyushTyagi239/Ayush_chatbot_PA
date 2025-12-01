"""
Ayush Personal RAG – FAST Answer Pipeline
- Uses local ChromaDB
- SentenceTransformer embeddings
- NVIDIA NIM or Groq for reranking + answering
- Optimized: Reranker only receives TOP 5 coarse chunks → 5× faster
"""

from dotenv import load_dotenv
from pathlib import Path
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from tenacity import retry, wait_exponential

# ================================
# ENV + CONFIG
# ================================
load_dotenv(override=True)

MODEL = "nvidia_nim/meta/llama-3.1-8b-instruct"

# Your REAL DB
DB_NAME = str(Path(__file__).resolve().parent.parent / "preprocessed_db")

# Your REAL collection inside DB (as confirmed by diagnostics)
collection_name = "docs"

# Retrieval
RETRIEVAL_K = 10      # coarse retrieval
RERANK_K = 5          # reranker input size (FAST)
FINAL_K = 5           # chunks sent to LLM answer

wait = wait_exponential(multiplier=1, min=2, max=60)

# Load local embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load Chroma client
chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(collection_name)


# ================================
# DATA MODELS
# ================================
class Result(BaseModel):
    page_content: str
    metadata: dict


# ================================
# RETRIEVAL
# ================================
def fetch_context_unranked(question):
    """embedding → Chroma → return K coarse chunks FAST"""
    print("\n🔍 Generating embedding for user question...")
    q_emb = embedder.encode([question], convert_to_numpy=True).tolist()[0]

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=RETRIEVAL_K,
        include=["documents", "metadatas"]   # SUPER IMPORTANT — faster!
    )

    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    for doc, meta in zip(docs, metas):
        chunks.append(Result(page_content=doc, metadata=meta))

    print(f"📚 Retrieved {len(chunks)} FAST chunks from Chroma.")

    return chunks



# ================================
# RERANKER
# ================================
@retry(wait=wait)
def rerank(question, chunks):
    print("\n🧠 Calling LLM to re-rank retrieved chunks...")

    sys_prompt = """
You are a chunk reranker.
Return ONLY JSON:  {"order":[1,2,3,...]}
No explanations.
"""

    user_text = f"User Question:\n{question}\n\nChunks:\n"
    for idx, c in enumerate(chunks):
        user_text += f"\nCHUNK ID: {idx+1}\n{c.page_content}\n"

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_text},
    ]

    response = completion(model=MODEL, messages=messages)
    raw = response.choices[0].message.content.strip()

    # parse safe JSON
    import json
    start = raw.find("{")
    end = raw.rfind("}")
    parsed = json.loads(raw[start:end+1])

    order = parsed["order"]
    print(f"📊 Re-ranking complete. Order: {order}")

    return [chunks[i - 1] for i in order]


# ================================
# MESSAGE BUILDER
# ================================
def make_rag_messages(question, chunks):
    context = "\n\n---\n\n".join(
        f"Source: {c.metadata['source']}\n{c.page_content}" for c in chunks
    )

    SYSTEM_PROMPT = f"""
You are a helpful assistant answering questions about Ayush Tyagi.
Use ONLY the provided context.
If answer is missing, say: "I don't have that information."

Context:
{context}
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


# ================================
# FINAL ANSWER
# ================================
@retry(wait=wait)
def answer_question(question: str, history: list = []):
    print("\n========================")
    print("   ⚡ NEW RAG REQUEST")
    print("========================")
    print(f"❓ User Question: {question}")

    # --------------------------------
    # 1) coarse retrieve
    # --------------------------------
    retrieved = fetch_context_unranked(question)

    # --------------------------------
    # ⭐ 2) LIMIT RERANKER INPUT TO TOP 5 → FAST MODE
    # --------------------------------
    retrieved_for_rerank = retrieved[:RERANK_K]

    # rerank only these 5
    reranked = rerank(question, retrieved_for_rerank)

    # --------------------------------
    # 3) select top FINAL_K chunks
    # --------------------------------
    final_chunks = reranked[:FINAL_K]
    print(f"📌 Using top {len(final_chunks)} chunks for final answer.")

    # --------------------------------
    # 4) build messages
    # --------------------------------
    messages = make_rag_messages(question, final_chunks)

    # --------------------------------
    # 5) LLM answer
    # --------------------------------
    print("🤖 Generating final answer...")
    response = completion(model=MODEL, messages=messages)
    answer = response.choices[0].message.content.strip()

    print("\n✅ Answer generated successfully!")
    return answer, final_chunks
