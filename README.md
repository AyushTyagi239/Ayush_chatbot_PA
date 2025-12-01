# Ayush Tyagi – Personal AI Assistant (RAG System)

A fully custom Retrieval-Augmented Generation (RAG) system designed to answer questions about Ayush Tyagi, built with modern, production-grade LLM engineering techniques.

This repository demonstrates an end-to-end RAG pipeline including chunking, embeddings, vector search, query rewriting, reranking, context assembly, and grounded answer generation.

---

## Overview

This assistant uses a multi-stage pipeline to:

- Load and preprocess knowledge  
- Generate structured LLM-based chunks  
- Store embeddings in a vector database  
- Retrieve relevant context  
- Rerank results  
- Produce answers strictly grounded in the knowledge base  

### Key Goals

- High accuracy  
- Low hallucination  
- Efficient retrieval  
- Consistent behavior  
- Scalable preprocessing  
- Strong evaluation flow  

---

## Architecture

Below is the main architecture diagram used in the project:

### System Diagram
<img width="832" height="1248" alt="Final_flow" src="https://github.com/user-attachments/assets/7d6e63ed-53f9-4d3c-83a8-1ba4ac990668" />


---

## Data Preparation Pipeline

This phase converts raw markdown files into high-quality vector embeddings.

Key steps include:

- Document loading (Markdown)  
- LLM-assisted chunking (headline, summary, content)  
- JSON validation using Pydantic  
- Embedding generation with MiniLM-L6-v2  
- Persistent storage using ChromaDB  
- Rich metadata storage for transparent retrieval  

---

## RAG Inference Pipeline

This phase handles real-time question answering using retrieval, reranking, and grounded generation.

Components include:

- Query rewriting for improved search intent  
- Vector similarity search (Top-K retrieval)  
- Reranking using NVIDIA NIM models  
- Context assembly with source-tagged chunks  
- Strict anti-hallucination prompting  
- Answer generation using NVIDIA NIM or Kimi models  
- Optional memory-based conversation continuation  
- Gradio-based user interface  

---

## Key Features

### LLM-Based Chunking  
Creates structured chunks with title, summary, and full text for optimal retrieval.

### Local Embeddings  
Uses MiniLM-L6-v2 to generate compact, efficient semantic embeddings.

### Chroma Vector Database  
Provides persistent and fast vector storage for accurate semantic search.

### Query Rewriting  
Improves retrieval accuracy by converting ambiguous inputs into optimized queries.

### Reranking  
Uses an LLM to reorder retrieved chunks based on relevance.

### Context-Grounded Responses  
Ensures all generated answers are strictly based on retrieved context.

### Evaluation Suite  
Tests the RAG pipeline on:

- Factual queries  
- Misspelled questions  
- Out-of-knowledge queries  
- Follow-up conversations  

Metrics included:

- Accuracy  
- Exact Match (EM)  
- Precision@K  
- Mean Reciprocal Rank (MRR)  
- Normalized Discounted Cumulative Gain (nDCG)  

### Gradio Web Interface  
Provides an interactive chat interface for testing and demonstrations.

---

## Evaluation

The evaluation module measures both retrieval quality and answer grounding across a variety of test categories.  
The insights from these tests help refine chunking quality, retrieval strategy, ranking accuracy, prompt design, and model behavior.

---

## Credits

This project is inspired by the RAG engineering techniques taught by Ed Donner.  
The design, structure, and methodology strongly follow principles demonstrated in his instructional content.

---

## Closing

This repository presents a complete, modular, and production-ready RAG system that integrates modern retrieval, rewriting, and reranking techniques.  
It functions as both a personal AI assistant for Ayush Tyagi and a practical demonstration of applied LLM engineering.

---
