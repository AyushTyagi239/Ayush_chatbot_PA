# answer.py — RAG pipeline (NO GPT references)
# Model-agnostic: set RAG_LLM_MODEL to your local / non-GPT model string
# Works with litellm completion, Chroma, HuggingFace embeddings

from pathlib import Path
from dotenv import load_dotenv
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from litellm import completion

load_dotenv(override=True)

# ---------------------------
# Config (no GPT names anywhere)
# ---------------------------
# Set this to any litellm-supported model string (local or hosted) that is NOT GPT.
# Examples you might use (examples only — pick what your litellm supports):
# - "mistral-7b" or "local-mistral" or "llama-2-13b" or a custom local path string
MODEL = os.getenv("RAG_LLM_MODEL", "local-model")  # <- set this in your .env

HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DB_DIR = os.getenv("RAG_DB_DIR", "vector_db")
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "ayush_rag")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))

# ---------------------------
# Embeddings + Vectorstore
# ---------------------------
embedding_fn = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_DIR,
    embedding_function=embedding_fn,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})


# ---------------------------
# Retrieval helper
# ---------------------------
def fetch_context(question: str) -> list[Document]:
    docs = retriever.invoke(question)
    if docs is None:
        return []
    if isinstance(docs, list):
        return docs
    return list(docs)


# ---------------------------
# Prompt builder (simple, safe)
# ---------------------------
def build_messages(question: str, context_text: str) -> list[dict]:
    system_msg = {
        "role": "system",
        "content": (
            "You are a RAG assistant. Use ONLY the provided context to answer. "
            "If the context doesn't contain the answer, reply: 'Information not available in knowledge-base.' "
            "Do not hallucinate."
        ),
    }

    user_msg = {
        "role": "user",
        "content": f"""Question:
{question}

Context:
{context_text}

Answer concisely and only using the above context."""
    }

    return [system_msg, user_msg]


# ---------------------------
# RAG answer (model-agnostic)
# ---------------------------
def answer_question(question: str) -> tuple[str, list[Document]]:
    # 1) Retrieve
    docs = fetch_context(question)

    # 2) Build context text
    if docs:
        chunks = []
        for d in docs:
            meta = d.metadata or {}
            title = meta.get("title") or meta.get("source") or ""
            header = f"[{title}]" if title else ""
            chunks.append(f"{header}\n{d.page_content}")
        context_text = "\n\n---\n\n".join(chunks)
    else:
        context_text = "No context available."

    # 3) Build messages
    messages = build_messages(question, context_text)

    # 4) Call the model via litellm (provider/model chosen by env var)
    resp = completion(model=MODEL, messages=messages)

    # 5) Extract answer safely
    answer_text = ""
    try:
        answer_text = resp.choices[0].message.content
    except Exception:
        try:
            answer_text = resp.choices[0].text
        except Exception:
            answer_text = str(resp)

    if not answer_text:
        answer_text = "No answer generated."

    return answer_text, docs


# ---------------------------
# CLI quick test
# ---------------------------
if __name__ == "__main__":
    q = input("Enter question: ").strip()
    ans, ctx = answer_question(q)
    print("\n========== ANSWER ==========")
    print(ans)
    print("\n========== TOP CONTEXT DOCS ==========")
    for i, d in enumerate(ctx, 1):
        print(f"\n[{i}] meta={d.metadata}")
        preview = (d.page_content[:400] + "...") if len(d.page_content) > 400 else d.page_content
        print(preview)
