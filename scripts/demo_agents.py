"""Demo script that wires Supervisor, DocumentAgent, and RetrieverAgent.

Shows ingestion and a sample retrieval via the orchestrator.
"""
import os
from src.agents.supervisor import Supervisor
from src.agents.document_agent import DocumentAgent
from src.agents.retriever_agent import RetrieverAgent


SAMPLE_TEXTS = [
    "Team Alpha SOW: implement a CRM connector and ETL pipelines.",
    "Bob: Data Engineer, skilled in Postgres and data pipelines.",
    "Case Study: Improved CRM data accuracy by 40% after migration."
]


def main():
    persist_dir = os.environ.get("RAG_PERSIST_DIR", "data/vector_store")
    sup = Supervisor()
    doc_agent = DocumentAgent(persist_dir=persist_dir)
    retr_agent = RetrieverAgent(persist_dir=persist_dir)

    sup.register("document", doc_agent)
    sup.register("retriever", retr_agent)

    # Ingest sample texts
    print("Ingesting sample texts...")
    res = sup.dispatch("document", {"type": "ingest_texts", "texts": SAMPLE_TEXTS})
    print("Ingest result:", res)

    # Query via retriever
    q = "Who is a Data Engineer?"
    print("Querying:", q)
    res = sup.dispatch("retriever", {"type": "query", "query": q, "k": 2})
    print("Query results:")
    for idx, r in enumerate(res.get("results", []), 1):
        print(f"{idx}. {r['content']}\n   metadata={r.get('metadata')}")


if __name__ == '__main__':
    main()
