# ---------------------------
# IMPORTS
# ---------------------------
from pathlib import Path

# LLM (Qwen via ChatOpenAI-compatible interface)
from langchain_openai import ChatOpenAI

# Vector DB (Chroma)
from langchain_chroma import Chroma

# HuggingFace Embeddings (local, no API calls)
from langchain_huggingface import HuggingFaceEmbeddings

# Core LangChain message/document types
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

# Load environment variables (API keys for LLM, not embeddings)
from dotenv import load_dotenv


# ---------------------------
# LOAD ENVIRONMENT VARIABLES
# ---------------------------
load_dotenv(override=True)


# ---------------------------
# CONSTANTS & MODEL SETUP
# ---------------------------

# Your main LLM (Qwen on NVIDIA NIM)
MODEL = "qwen/qwen3-next-80b-a3b-thinking"

# Path to your ChromaDB directory
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

# Local embedding model (NO API → NO 404 errors)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Number of retrieved chunks for each query
RETRIEVAL_K = 5


# ---------------------------
# SYSTEM PROMPT (RAG instructions)
# ---------------------------
SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.

Use the following context when relevant.
If the answer is not in the context, say you don't know.

Context:
{context}
"""


# ---------------------------
# INITIALIZE VECTORSTORE + RETRIEVER + LLM
# ---------------------------

# Load the persistent Chroma DB
vectorstore = Chroma(
    persist_directory=DB_NAME,
    embedding_function=embeddings
)

# Convert Chroma into a retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": RETRIEVAL_K}
)

# Qwen LLM (via ChatOpenAI wrapper, works with NVIDIA API)
llm = ChatOpenAI(
    temperature=0,
    model_name=MODEL  # litellm handles API calls under the hood
)


# ---------------------------
# FUNCTION: Retrieve relevant context documents
# ---------------------------
def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant documents using pure vector search (NO LLM call).
    """
    #return retriever.get_relevant_documents(question)
    return retriever.invoke(question)



# ---------------------------
# FUNCTION: Combine past history with new question
# ---------------------------
def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine past user messages into a single long text
    to improve retrieval accuracy.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question if prior else question


# ---------------------------
# FUNCTION: Main RAG answer pipeline
# ---------------------------
def answer_question(
    question: str,
    history: list[dict] = []
) -> tuple[str, list[Document]]:
    """
    Steps:
    1. Combine conversation history with new question
    2. Retrieve relevant context documents
    3. Insert context into system prompt
    4. Call Qwen LLM with system prompt + history + user question
    5. Return final answer + retrieved docs
    """

    # 1. Build retrieval text from history + question
    combined = combined_question(question, history)

    # 2. Retrieve matching documents
    docs = fetch_context(combined)

    # 3. Create context string for prompt
    context = "\n\n".join(doc.page_content for doc in docs)

    # 4. Build system message
    system_prompt = SYSTEM_PROMPT.format(context=context)

    # 5. Construct chat messages
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))

    # 6. Call Qwen
    response = llm.invoke(messages)

    # Return the text response + docs used
    return response.content, docs
