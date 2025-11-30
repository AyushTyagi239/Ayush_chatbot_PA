#!/usr/bin/env python3
"""
Ayush Personal RAG Pipeline (NVIDIA NIM LLM + local embeddings)

- Chunking: uses NVIDIA NIM model to split documents into overlapping chunks
  (we ask for JSON but must tolerate raw text; we repair if needed).
- Embeddings: created locally using sentence-transformers (all-MiniLM-L6-v2).
- Vector DB: uses Chroma (PersistentClient).
- Retrieval: queries Chroma with local embeddings.
- Reranker: uses NIM to reorder retrieved chunks (model returns JSON-ish text; we parse/repair).
- Answering: final LLM call to generate response using top chunks.

Notes:
- Install required libs: pip install litellm chromadb sentence-transformers tqdm tenacity pydantic
- Ensure NVIDIA_API_KEY (or appropriate LLM credentials) are set in your environment if using NIM.
"""

from pathlib import Path
import json
import re
import os
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from multiprocessing import Pool
from tenacity import retry, wait_exponential, stop_after_attempt
from sentence_transformers import SentenceTransformer

# -------------------------
# CONFIG
# -------------------------
load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# LLM model to use for chunking / reranking / answering (NVIDIA NIM example)
MODEL = "nvidia_nim/meta/llama-3.1-8b-instruct"

# Local DB path for Chroma (persistent)
DB_NAME = str(Path(__file__).resolve().parent.parent / "vector_db")
collection_name = "docs"


# Local embedding model name (runs locally)
LOCAL_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Knowledge base path (assumes script is in pro_implementation/ and knowledge-base is sibling)
KNOWLEDGE_BASE_PATH = Path(__file__).resolve().parent.parent / "knowledge-base"

# How many characters target per chunk (estimation used for prompt)DB_NAME
AVERAGE_CHUNK_SIZE = 300

# Multiprocessing workers for chunk generation
WORKERS = 4

# Retry policy for LLM calls: exponential backoff, up to 3 attempts
wait = wait_exponential(multiplier=1, min=2, max=30)
stop = stop_after_attempt(3)

# Retrieval parameter: how many chunks to fetch
RETRIEVAL_K = 10

# -------------------------
# MODELS / CLIENTS
# -------------------------
# Local embedding model (no API key required)
embedder = SentenceTransformer(LOCAL_EMBED_MODEL_NAME)

# Note: create PersistentClient inside functions to avoid multiprocessing issues.

# -------------------------
# DATA MODELS
# -------------------------


class Result(BaseModel):
    """Object stored in Chroma and returned by retrieval functions."""
    page_content: str
    metadata: dict


class Chunk(BaseModel):
    """Chunk representation produced by LLM chunker (validated via pydantic)."""
    headline: str = Field(description="Short heading for this chunk")
    summary: str = Field(description="2-4 sentence summary of the chunk")
    original_text: str = Field(description="Exact original text for the chunk")

    def as_result(self, document):
        """Convert to Result with metadata from original document."""
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )


class Chunks(BaseModel):
    """Wrapper used to validate parsed JSON: { "chunks": [ {headline, summary, original_text}, ... ] }"""
    chunks: list[Chunk]


# -------------------------
# JSON extraction / repair helpers
# -------------------------


def extract_json_from_text(text: str):
    """
    Try to extract the first JSON object from text and parse it.
    If direct parsing fails, perform minimal safe repairs (quotes, trailing commas).
    """
    # locate first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output.")
    json_text = text[start:end + 1]

    # try plain load
    try:
        return json.loads(json_text)
    except Exception:
        pass

    # fallback repairs
    alt = json_text

    # 1) Replace single quotes with double quotes when it's probably safe
    alt = alt.replace("'", '"')

    # 2) Remove trailing commas before } or ]
    alt = re.sub(r",\s*([}\]])", r"\1", alt)

    # 3) Ensure embedded newlines are escaped (only inside strings heuristically)
    # This is aggressive but often necessary: replace raw newlines between quotes with \n
    def escape_newlines_in_strings(s):
        out = []
        in_str = False
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == '"' and (i == 0 or s[i - 1] != "\\"):
                in_str = not in_str
                out.append(ch)
                i += 1
            elif in_str and ch == "\n":
                out.append("\\n")
                i += 1
            else:
                out.append(ch)
                i += 1
        return "".join(out)

    alt = escape_newlines_in_strings(alt)

    # Try parsing again
    try:
        return json.loads(alt)
    except Exception as e:
        # As last resort, attempt a second-level repair focusing on original_text fields:
        # find "original_text": " ... " patterns and escape inner quotes
        try:
            def escape_inner_quotes(match):
                content = match.group(1)
                # escape double quotes and backslashes inside content
                content = content.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                return f'"original_text":"{content}"'
            alt2 = re.sub(r'"original_text"\s*:\s*"([^"]*?)"', escape_inner_quotes, alt, flags=re.DOTALL)
            alt2 = re.sub(r",\s*([}\]])", r"\1", alt2)
            return json.loads(alt2)
        except Exception:
            # If still failing, raise with helpful debug
            raise ValueError("Failed to parse JSON after repair attempts.") from e


def repair_and_extract(raw: str):
    """
    Try parsing JSON directly; if it fails, attempt repairs and return parsed object.
    Returns Python dict.
    """
    try:
        return extract_json_from_text(raw)
    except Exception as e:
        logging.debug("Initial JSON parse failed; attempting repair.")
        # Heuristic repairs: replace CRLF, fix common bad sequences
        candidate = raw.replace("\r\n", "\n").replace("\r", "\n")
        candidate = candidate.replace("\t", "    ")
        # Remove weird control characters
        candidate = "".join(c for c in candidate if ord(c) >= 9)
        try:
            return extract_json_from_text(candidate)
        except Exception:
            # final attempt: escape raw newlines globally inside the top-level braces slice
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start == -1 or end == -1:
                raise
            payload = candidate[start:end + 1]
            payload_escaped = payload.replace("\n", "\\n").replace('\\"', '\\\\"')
            try:
                return json.loads(payload_escaped)
            except Exception as final_e:
                logging.error("All repair attempts failed.")
                logging.debug("RAW OUTPUT (first 2000 chars):\n%s", raw[:2000])
                raise final_e from e


# -------------------------
# 1) DOCUMENT LOADING
# -------------------------


def fetch_documents():
    """
    Load all markdown documents under KNOWLEDGE_BASE_PATH.
    Each first-level folder name becomes the document 'type'.
    Returns a list of dicts: { "type": <folder>, "source": <filepath>, "text": <content> }.
    """
    documents = []
    if not KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError(f"Knowledge base path not found: {KNOWLEDGE_BASE_PATH}")

    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        if not folder.is_dir():
            continue
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                documents.append({
                    "type": doc_type,
                    "source": file.as_posix(),
                    "text": f.read()
                })
    logging.info("Loaded %d documents", len(documents))
    return documents


# -------------------------
# 2) PROMPT + LLM CHUNKING
# -------------------------

# A strict system prompt asking the model to return JSON only.
JSON_SAFE_CHUNKER_PROMPT = """
You MUST output valid JSON only.

CRITICAL RULES:
1. Escape ALL newline characters inside strings as "\\n" if you'll include raw text in a string.
2. Escape ALL double quotes inside strings as \\".
3. DO NOT output markdown blocks unescaped. (If you must include markdown, ensure newlines are \\n and quotes are escaped.)
4. Output EXACTLY this structure (no additional keys, no comments):

{
  "chunks": [
    {
      "headline": "...",
      "summary": "...",
      "original_text": "..."
    }
  ]
}

5. "original_text" must contain the original text (can be escaped with \\n and \\"), but must be a valid JSON string.
6. Do NOT output anything before or after the JSON block.
7. Do NOT invent text. Do NOT omit content.
8. NEVER break JSON formatting.
"""

def escape_document_text_for_prompt(text: str) -> str:
    """
    Escape document text before embedding into the prompt to reduce LLM mistakes.
    We escape backslashes, quotes and newlines to ask the LLM to preserve the content but output JSON-safe strings.
    """
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def make_prompt(document):
    """
    JSON-safe prompt for chunking. We include the escaped document text inside quotes so the model sees it as an escaped string.
    """
    how_many = max(1, (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1)
    safe_text = escape_document_text_for_prompt(document["text"])

    return f"""{JSON_SAFE_CHUNKER_PROMPT}

Split the following (escaped) document into overlapping JSON chunks.

Document type: {document['type']}
Source: {document['source']}
Approx chunk count (guide): {how_many}

Document (escaped; contains \\n for newlines):
"{safe_text}"
"""

# Retry LLM chunking up to 3 times with exponential backoff
@retry(wait=wait, stop=stop)
def process_document(document):
    """
    Send a single document to the LLM to create JSON chunks.
    - Calls the LLM via litellm.completion
    - Repairs & parses returned JSON robustly
    - Validates with pydantic (Chunks) and returns a list of Result objects
    """
    messages = [{"role": "user", "content": make_prompt(document)}]

    # Call the LLM (NIM)
    response = completion(model=MODEL, messages=messages)
    raw = response.choices[0].message.content.strip()

    # Try to parse/repair JSON from the model output
    try:
        parsed = repair_and_extract(raw)
    except Exception as e:
        logging.error("Failed to parse/recover JSON for document %s", document["source"])
        logging.debug("RAW OUTPUT (first 2000 chars):\n%s", raw[:2000])
        raise

    # Validate with Pydantic
    try:
        chunks_obj = Chunks.model_validate(parsed)
    except Exception as e:
        logging.error("Pydantic validation failed for parsed JSON from %s", document["source"])
        logging.debug("Parsed JSON (compact): %s", json.dumps(parsed)[:2000])
        raise

    # Convert Chunks -> Result objects (page_content + metadata)
    results = [c.as_result(document) for c in chunks_obj.chunks]
    return results


def create_chunks(documents):
    """
    Sequential chunk creation (SAFE).
    Multiprocessing breaks because LLM objects cannot be pickled.
    """
    chunks = []
    for doc in tqdm(documents, desc="Chunking"):
        try:
            result = process_document(doc)  # direct call
            chunks.extend(result)
        except Exception as e:
            print(f"\n[ERROR] Failed to process document: {doc['source']}")
            print(e)
            continue
    return chunks



# -------------------------
# 3) EMBEDDING + CHROMA STORAGE
# -------------------------


def create_embeddings(chunks):
    """
    Given a list of Result objects (each with page_content and metadata), compute local embeddings
    and add them to a Chroma collection. Recreates the collection if it exists.
    """
    chroma = PersistentClient(path=DB_NAME)

    # Delete existing collection if present
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    # Prepare data
    texts = [c.page_content for c in chunks]
    metas = [c.metadata for c in chunks]
    ids = [str(i) for i in range(len(chunks))]

    # Compute local embeddings (SentenceTransformers)
    logging.info("Computing local embeddings (SentenceTransformers)...")
    vectors = embedder.encode(texts, convert_to_numpy=True).tolist()

    # Create collection and add data
    collection = chroma.get_or_create_collection(collection_name)
    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)

    logging.info("Vectorstore created with %d documents", collection.count())


# -------------------------
# 4) RETRIEVAL
# -------------------------


def fetch_context_unranked(question, k=RETRIEVAL_K):
    """
    Convert a user question into embedding (locally) and query Chroma for top-k documents.
    Returns a list of Result objects (page_content + metadata) in the order returned by Chroma.
    """
    chroma = PersistentClient(path=DB_NAME)
    collection = chroma.get_or_create_collection(collection_name)

    # Compute question embedding locally
    q_vec = embedder.encode([question], convert_to_numpy=True).tolist()[0]

    # Query Chroma
    results = collection.query(query_embeddings=[q_vec], n_results=k)

    # Convert results to Result objects
    docs = [Result(page_content=d, metadata=m) for d, m in zip(results["documents"][0], results["metadatas"][0])]
    return docs


# -------------------------
# 5) RERANKER (LLM)
# -------------------------


def rerank(question, chunks):
    """
    Use the LLM to rerank a provided list of chunks for relevance to the question.
    Expects a JSON-like response: { "order": [id1, id2, ...] }
    Returns the chunks reordered by relevance (most relevant first).
    """
    system_prompt = """
You are a chunk reranker. You will receive a user question and a list of text chunks with chunk IDs.
Return ONLY a JSON object: { "order": [id1, id2, ...] } listing all chunk ids in order most->least relevant.
No extra commentary.
"""
    user_prompt = f"Question:\n{question}\n\nChunks:\n"
    for idx, c in enumerate(chunks):
        user_prompt += f"CHUNK ID: {idx + 1}\n{c.page_content}\n\n"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    response = completion(model=MODEL, messages=messages)
    raw = response.choices[0].message.content.strip()

    try:
        parsed = repair_and_extract(raw)
        order = parsed.get("order")
        if not isinstance(order, list):
            raise ValueError("Parsed JSON does not contain 'order' list")
    except Exception:
        logging.error("Reranker output parse failed. Raw (truncated): %s", raw[:2000])
        raise

    # convert 1-based ids to 0-based indices and return ordered chunks
    ordered_chunks = [chunks[i - 1] for i in order]
    return ordered_chunks


# -------------------------
# 6) FINAL ANSWER GENERATION
# -------------------------


SYSTEM_PROMPT = """
You are a helpful, concise assistant that answers questions about Ayush Tyagi using only the provided Knowledge Base extracts.
If the answer is not in the provided context, say "I don't have that information."
Be accurate, concise, and avoid inventing facts.
Context:
{context}
"""


def answer_question(question, history=None):
    """
    Full RAG answer flow:
    - fetch_context_unranked -> rerank -> build final context -> call LLM to answer
    Returns final text answer and the docs used.
    """
    if history is None:
        history = []

    # 1) retrieve unranked top-K chunks
    retrieved = fetch_context_unranked(question, k=RETRIEVAL_K)

    # 2) rerank with LLM (improves ordering)
    try:
        reranked = rerank(question, retrieved)
    except Exception:
        # If reranker fails, fall back to unranked results
        logging.warning("Reranker failed — falling back to unranked retrieval order.")
        reranked = retrieved

    # 3) build context using top N chunks (e.g., top 5)
    top_k_for_answer = min(5, len(reranked))
    top_chunks = reranked[:top_k_for_answer]
    context_text = "\n\n---\n\n".join(c.page_content for c in top_chunks)

    # 4) call LLM to generate the answer (we ask NIM to output plain text)
    system_prompt = SYSTEM_PROMPT.format(context=context_text)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
    response = completion(model=MODEL, messages=messages)
    answer = response.choices[0].message.content.strip()

    return answer, top_chunks


# -------------------------
# 7) MAIN (INGESTION + QUICK TEST)
# -------------------------


if __name__ == "__main__":
    # 1) Load documents
    documents = fetch_documents()

    # 2) Chunk documents (LLM)
    chunks = create_chunks(documents)

    # 3) Create embeddings + save to Chroma
    create_embeddings(chunks)

    # 4) Quick test query
    test_q = "Which year did Ayush Tyagi complete his B.Tech?"
    ans, used = answer_question(test_q)
    print("\n=== Test Query ===")
    print("Question:", test_q)
    print("Answer:", ans)
    print("Chunks used:")
    for i, c in enumerate(used):
        print(i, c.metadata, c.page_content[:200].replace("\n", " "), "...")
