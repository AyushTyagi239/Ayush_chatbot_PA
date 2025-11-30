"""
Ayush Personal RAG – Answer Pipeline
- Uses local ChromaDB
- Local embeddings (SentenceTransformers)
- NVIDIA NIM or Groq LLM for reranking + final answer
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

MODEL = "nvidia_nim/meta/llama-3.1-8b-instruct"   # or your Groq model
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
collection_name = "ayush_docs"

RETRIEVAL_K = 15
FINAL_K = 5

wait = wait_exponential(multiplier=1, min=2, max=60)

# Load local embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load ChromaDB
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
    """Convert question → embedding → query Chroma → return top-K chunks."""
    print("\n🔍 Generating embedding for user question...")
    q_emb = embedder.encode([question], convert_to_numpy=True).tolist()[0]

    results = collection.query(query_embeddings=[q_emb], n_results=RETRIEVAL_K)

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=doc, metadata=meta))

    print(f"📚 Retrieved {len(chunks)} unranked chunks from Chroma.")
    return chunks


# ================================
# RERANKER (LLM)
# ================================
@retry(wait=wait)
def rerank(question, chunks):
    """Ask LLM to reorder chunks by relevance."""
    print("\n🧠 Calling LLM to re-rank retrieved chunks...")

    sys_prompt = """
You are a chunk reranker.
Return ONLY a JSON object: { "order": [1,2,3,...] }
List all IDs from most relevant to least relevant.
Do NOT add explanations.
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

    # Extract JSON safely
    try:
        import json
        start = raw.find("{")
        end = raw.rfind("}")
        parsed = json.loads(raw[start:end+1])
    except Exception:
        print("\n❌ Failed to parse reranker output:\n", raw)
        raise

    order = parsed["order"]

    print(f"📊 Re-ranking complete. Order: {order}")
    return [chunks[i - 1] for i in order]


# ================================
# MESSAGE BUILDER
# ================================
def make_rag_messages(question, chunks):
    """Build system + user messages for final answer."""
    context = "\n\n---\n\n".join(
        f"Source: {c.metadata['source']}\n{c.page_content}"
        for c in chunks
    )

    SYSTEM_PROMPT = f"""
You are a helpful assistant answering questions about Ayush Tyagi.
Use ONLY the provided context.
If the answer is not found, say: "I don't have that information."

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
    """
    Full RAG flow:
    1. Retrieve similar chunks
    2. Re-rank them using LLM
    3. Build a context window
    4. Generate final answer
    """
    print("\n========================")
    print("   ⚡ NEW RAG REQUEST")
    print("========================")
    print(f"❓ User Question: {question}")

    # 1) Retrieve
    retrieved = fetch_context_unranked(question)

    # 2) Rerank
    reranked = rerank(question, retrieved)

    # 3) Select top FINAL_K chunks
    final_chunks = reranked[:FINAL_K]

    print(f"📌 Using top {len(final_chunks)} chunks for final answer.")

    # 4) Prepare messages
    messages = make_rag_messages(question, final_chunks)

    # 5) Call LLM
    print("🤖 Generating final answer...")
    response = completion(model=MODEL, messages=messages)
    answer = response.choices[0].message.content.strip()

    print("\n✅ Answer generated successfully!")
    return answer, final_chunks
