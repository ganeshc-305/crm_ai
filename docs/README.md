CRM AI Assistant — Documentation

Overview

This project is a prototype CRM assistant using Streamlit, LangChain, and a RAG pipeline.

Key components
- streamlit_app.py — UI for uploads, ingestion, and chat
- src/rag — ingestion, retrieval, and loaders
- src/agents — simple in-process multi-agent orchestrator
- src/db.py — optional PostgreSQL integration (SQLAlchemy)

Quickstart
1. Create a .env based on .env.example and set OPENAI_API_KEY and DATABASE_URL (optional).
2. Install dependencies: pip install -r requirements.txt
3. Run ingestion: python scripts/ingest_files.py --dir data/uploads
4. Run Streamlit: streamlit run streamlit_app.py

Testing
See docs/TESTS.md for running unit tests and target areas to validate.
