#!/usr/bin/env python3
"""
Ayush Personal FAST RAG Ingestion Pipeline
- Uses NVIDIA NIM for safe JSON chunking
- Uses local MiniLM embeddings
- Writes into Chroma: preprocessed_db/docs
"""

from pathlib import Path
import json, re, logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from tenacity import retry, wait_exponential, stop_after_attempt
from sentence_transformers import SentenceTransformer

# =================================================
# CONFIG
# =================================================
load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

MODEL = "nvidia_nim/meta/llama-3.1-8b-instruct"

# IMPORTANT → Use your real DB
DB_NAME = str(Path(__file__).resolve().parent.parent / "preprocessed_db")
collection_name = "docs"

# Local MiniLM embedding model
LOCAL_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Knowledge base folder
KNOWLEDGE_BASE_PATH = Path(__file__).resolve().parent.parent / "knowledge-base"

AVERAGE_CHUNK_SIZE = 300
RETRIEVAL_K = 10

wait = wait_exponential(multiplier=1, min=2, max=30)
stop = stop_after_attempt(3)

# Embedding model
embedder = SentenceTransformer(LOCAL_EMBED_MODEL_NAME)

# =================================================
# MODELS
# =================================================

class Result(BaseModel):
    page_content: str
    metadata: dict

class Chunk(BaseModel):
    headline: str
    summary: str
    original_text: str

    def as_result(self, document):
        return Result(
            page_content=f"{self.headline}\n\n{self.summary}\n\n{self.original_text}",
            metadata={"source": document["source"], "type": document["type"]},
        )

class Chunks(BaseModel):
    chunks: list[Chunk]

# =================================================
# JSON Repair Helpers
# =================================================

def extract_json_from_text(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found")
    raw = text[start:end + 1]

    try:
        return json.loads(raw)
    except Exception:
        raw = raw.replace("'", '"')
        raw = re.sub(r",\s*([}\]])", r"\1", raw)
        return json.loads(raw)

def repair_and_extract(raw: str):
    try:
        return extract_json_from_text(raw)
    except Exception:
        raw = raw.replace("\r", "").replace("\t", " ")
        return extract_json_from_text(raw)

# =================================================
# DOCUMENT LOADING
# =================================================

def fetch_documents():
    documents = []

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

    logging.info(f"Loaded {len(documents)} documents")
    return documents

# =================================================
# LLM CHUNKING
# =================================================

JSON_SAFE_CHUNKER_PROMPT = """
You MUST output valid JSON only.
Correct format:

{"chunks":[{"headline":"...", "summary":"...", "original_text":"..."}]}
"""

def escape_document_text_for_prompt(text):
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

def make_prompt(document):
    safe = escape_document_text_for_prompt(document["text"])
    return f"""
{JSON_SAFE_CHUNKER_PROMPT}

Document type: {document['type']}
Source: {document['source']}

Document:
"{safe}"
"""

@retry(wait=wait, stop=stop)
def process_document(document):
    response = completion(model=MODEL, messages=[{
        "role": "user",
        "content": make_prompt(document)
    }])

    raw = response.choices[0].message.content

    parsed = repair_and_extract(raw)
    chunk_obj = Chunks.model_validate(parsed)

    return [c.as_result(document) for c in chunk_obj.chunks]

def create_chunks(documents):
    all_chunks = []
    for doc in tqdm(documents, desc="Chunking"):
        try:
            all_chunks.extend(process_document(doc))
        except Exception as e:
            print("Failed:", doc["source"])
            continue
    return all_chunks

# =================================================
# EMBEDDINGS + CHROMA
# =================================================

def create_embeddings(chunks):
    chroma = PersistentClient(path=DB_NAME)

    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    collection = chroma.get_or_create_collection(collection_name)

    texts = [c.page_content for c in chunks]
    metas = [c.metadata for c in chunks]
    ids = [str(i) for i in range(len(chunks))]

    logging.info("Encoding embeddings...")
    vectors = embedder.encode(texts, convert_to_numpy=True).tolist()

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)

    logging.info(f"Vectorstore ready! Total chunks: {collection.count()}")

# =================================================
# MAIN
# =================================================

if __name__ == "__main__":
    docs = fetch_documents()
    chunks = create_chunks(docs)
    create_embeddings(chunks)
    print("\nPreprocessing complete!")
