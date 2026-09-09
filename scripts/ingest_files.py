"""Ingest files from a directory into the vector store.

Usage:
  python scripts/ingest_files.py --dir data/uploads --persist data/vector_store
"""
import os
import argparse
import logging

from src.rag.loaders import load_documents_from_dir, split_documents_to_texts
from src.rag.ingest import create_vectorstore_from_texts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=os.environ.get("UPLOAD_DIR", "data/uploads"), help="Directory with files to ingest")
    p.add_argument("--persist", default=os.environ.get("RAG_PERSIST_DIR", "data/vector_store"), help="Vector store persist dir")
    p.add_argument("--chunk_size", type=int, default=1000)
    p.add_argument("--chunk_overlap", type=int, default=200)
    args = p.parse_args()

    documents = load_documents_from_dir(args.dir)
    if not documents:
        logger.info("No documents found to ingest in %s", args.dir)
        return
    texts, metadatas = split_documents_to_texts(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    logger.info("Ingesting %d chunks to %s", len(texts), args.persist)
    vs = create_vectorstore_from_texts(texts, metadatas=metadatas, persist_dir=args.persist)
    logger.info("Ingestion completed. Vector store type: %s", type(vs).__name__)
    print({"status": "ok", "ingested_chunks": len(texts), "vectorstore": type(vs).__name__})

if __name__ == '__main__':
    main()
