"""Small demo: create a vectorstore from sample texts and run a query.

Usage: python scripts/demo_rag.py
"""
import os
from src.rag.ingest import create_vectorstore_from_texts
from src.rag.retrieve import RAGRetriever


SAMPLE_TEXTS = [
    "Project X SOW: build a CRM integration using REST APIs and PostgreSQL.",
    "Alice: Senior Engineer with skills in Python, LangChain and embeddings.",
    "Project Y Case Study: migrated legacy CRM to cloud, improved data quality.",
    "Employee resume: experienced in data pipelines, Postgres, and Streamlit UIs."
]


def main():
    persist_dir = os.environ.get("RAG_PERSIST_DIR", "data/vector_store")
    print("Creating vectorstore from sample texts (this may call the embedding API)...")
    vs = create_vectorstore_from_texts(SAMPLE_TEXTS, persist_dir=persist_dir)
    print("Done. Loading retriever...")
    retr = RAGRetriever(persist_dir=persist_dir)
    if not retr.is_ready():
        print("Retriever not ready. Vector store not found or failed to load.")
        return
    q = "Who is skilled in LangChain?"
    print(f"Query: {q}")
    docs = retr.query(q, k=3)
    print("Top results:")
    for i, d in enumerate(docs, 1):
        print(f"{i}. {d.page_content}\n   metadata={d.metadata}\n")


if __name__ == '__main__':
    main()
