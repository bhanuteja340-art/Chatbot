# AI Interview Assistant

This is a simple RAG-based interview assistant.

It uses an interview questions and answers PDF as the knowledge source. When a user asks a question, the application searches the relevant content from the PDF and uses Gemini to generate an answer.

## How it works

User Question
      ↓
Query Embedding
      ↓
ChromaDB Search
      ↓
Top 5 Relevant Chunks
      ↓
Gemini
      ↓
Answer

## Project Structure


Chatbot/
│
├── knowledge_base/
│   └── interview questions PDF
│
├── src/
│   ├── pdf_loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── ingest.py
│   ├── retriever.py
│   ├── generator.py
│   └── rag_pipeline.py
│
├── vector_store/
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore

## Setup

Install the required packages:

pip install -r requirements.txt

## Run

For the command-line application: python main.py

For the web application: For the web application: