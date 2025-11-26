import os
import glob
from pathlib import Path

# Load markdown files recursively
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Splits documents into RAG-friendly chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector database storage
from langchain_chroma import Chroma

# Embedding model (OpenAI recommended)
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv


# -----------------------------
# CONFIGURATION
# -----------------------------

# LLM model (used later in retrieval pipeline)
MODEL = "qwen/qwen3-next-80b-a3b-instruct"

# Path to vector database directory
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

# Path to your knowledge-base folder
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

# Load API keys from .env
load_dotenv(override=True)

# Use OpenAI embeddings for vectorization
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# -----------------------------
# LOAD DOCUMENTS
# -----------------------------
def fetch_documents():
    """
    Load all markdown files from every folder inside /knowledge-base.
    Also attach metadata like folder name (doc_type).
    """
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []

    for folder in folders:
        doc_type = os.path.basename(folder)  # e.g. "skills", "experience"

        # Load all .md files inside this folder
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )

        folder_docs = loader.load()

        # Add metadata so you know where the doc came from
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)

    return documents


# -----------------------------
# CHUNK DOCUMENTS
# -----------------------------
def create_chunks(documents):
    """
    Split long documents into smaller 500-character chunks.
    Overlap (200 chars) keeps context intact.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


# -----------------------------
# EMBED & STORE IN CHROMA DB
# -----------------------------
def create_embeddings(chunks):
    """
    Convert chunks into embeddings and save them in ChromaDB.
    Deletes existing DB before rebuilding.
    """

    # Wipe old database if exists
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    # Build vector store from chunks
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME
    )

    # For debugging: show statistics
    collection = vectorstore._collection
    count = collection.count()
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)

    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore


# -----------------------------
# MAIN SCRIPT ENTRY POINT
# -----------------------------
if __name__ == "__main__":

    # 1. Load markdown docs
    documents = fetch_documents()

    # 2. Convert into RAG chunks
    chunks = create_chunks(documents)

    # 3. Embed & store in ChromaDB
    create_embeddings(chunks)

    print("Ingestion complete")
