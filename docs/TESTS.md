Running tests

Tests use pytest. Install dev deps then run:

    pip install -r requirements.txt
    pytest -q

Test coverage
- tests/test_loaders.py: validates file loading and chunking (uses temp files).
- tests/test_db.py: initializes an in-memory SQLite DB, tests save_message/load_recent_messages.
- tests/test_supervisor.py: tests Supervisor.register/dispatch with a mock agent.

Notes
- Tests avoid network calls and OpenAI by using SQLite and local file IO.
- Add more tests for RAG ingestion and retriever using mocking if desired.
