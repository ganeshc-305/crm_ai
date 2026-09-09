import os
from src import db as db_mod


def test_init_and_save_load_messages():
    # Use SQLite in-memory for tests
    engine = db_mod.init_db('sqlite:///:memory:')
    assert engine is not None

    res = db_mod.save_message('user', 'hello test', metadata={'test': True})
    assert res['role'] == 'user'
    assert 'id' in res

    rows = db_mod.load_recent_messages(limit=10)
    assert len(rows) >= 1
    assert rows[-1]['content'] == 'hello test'
