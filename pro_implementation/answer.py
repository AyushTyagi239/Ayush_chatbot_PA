"""
Ayush Personal RAG – FAST Answer Pipeline
"""

from dotenv import load_dotenv
from pathlib import Path
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from tenacity import retry, wait_exponential
import json

# ============================================
# CONFIG
# ============================================
load_dotenv(override=True)

MODEL = "nvidia_nim/meta/llama-3.1-8b-instruct"

DB_NAME = str(Path(__file__).resolve().parent.parent / "preprocessed_db")
collection_name = "docs"

RETRIEVAL_K = 10
RERANK_K = 5
FINAL_K = 5

wait = wait_exponential(multiplier=1, min=2, max=60)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(collection_name)

# ============================================
# MODEL OBJECTS
# ============================================
class Result(BaseModel):
    page_content: str
    metadata: dict

# ============================================
# RETRIEVAL
# ============================================
def fetch_context_unranked(question):
    print("\n🔍 Embedding question...")
    q = embedder.encode([question], convert_to_numpy=True).tolist()[0]

    results = collection.query(
        query_embeddings=[q],
        n_results=RETRIEVAL_K,
        include=["documents", "metadatas"]
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    out = [Result(page_content=d, metadata=m) for d, m in zip(docs, metas)]
    print(f"📚 Retrieved {len(out)} chunks.")
    return out

# ============================================
# RERANKER
# ============================================
@retry(wait=wait)
def rerank(question, chunks):
    print("\n🧠 Reranking top chunks...")
    sys_prompt = """
Return JSON only: {"order":[1,2,3,...]}
"""

    user_text = f"User Question:\n{question}\n\nChunks:\n"
    for i, ch in enumerate(chunks):
        user_text += f"\nCHUNK {i+1}:\n{ch.page_content}\n"

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_text}
    ]

    resp = completion(model=MODEL, messages=messages)
    raw = resp.choices[0].message.content

    start, end = raw.find("{"), raw.rfind("}")
    order = json.loads(raw[start:end+1])["order"]

    return [chunks[i-1] for i in order]

# ============================================
# MESSAGE BUILDER
# ============================================
def make_rag_messages(question, chunks):
    ctx = "\n\n---\n\n".join(
        f"Source: {c.metadata['source']}\n{c.page_content}"
        for c in chunks
    )

    sys_prompt = f"""
You answer ONLY using the provided context.
If info is missing, say: "I don't have that information."

Context:
{ctx}
"""

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question},
    ]

# ============================================
# ANSWER
# ============================================
@retry(wait=wait)
def answer_question(question: str, history=[]):
    print("\n===============")
    print("⚡ NEW REQUEST")
    print("===============")
    print("❓:", question)

    retrieved = fetch_context_unranked(question)

    reranked = rerank(question, retrieved[:RERANK_K])

    final_chunks = reranked[:FINAL_K]

    messages = make_rag_messages(question, final_chunks)
    resp = completion(model=MODEL, messages=messages)
    answer = resp.choices[0].message.content.strip()

    print("\n✅ Done.")
    return answer, final_chunks
