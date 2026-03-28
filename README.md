# PDF RAG Chatbot

A Retrieval-Augmented Generation chatbot that answers questions from uploaded PDFs.

## User Interface

![Chatbot Demo](assets/UI.png)

## System Architecture

User Query  
   ↓  
Streamlit Chat Interface  
   ↓  
Retriever (FAISS)  
   ↓  
Relevant PDF Chunks  
   ↓  
Prompt + Context  
   ↓  
Ollama LLM  
   ↓  
Answer + Source Citation

## Features
- Multi-PDF support
- FAISS vector database
- HuggingFace embeddings
- Ollama LLM
- Source citation

## Tech Stack
- Python
- Streamlit
- LangChain
- FAISS
- Ollama

## How to Run

pip install -r requirements.txt
streamlit run app.py
