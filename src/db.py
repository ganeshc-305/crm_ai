"""PostgreSQL integration helpers using SQLAlchemy.

Usage:
- Call init_db(database_url) at startup (database_url optional). If provided, tables are created.
- Use save_message(role, content, metadata) to persist chat messages.
- Use load_recent_messages(limit) to retrieve history.

Supports converting JDBC-style URLs (jdbc:postgresql://host:port/db) to SQLAlchemy driver format.
"""
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()
_engine = None
_SessionLocal = None


class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    id = Column(Integer, primary_key=True, autoincrement=True)
    role = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    message_metadata = Column('metadata', JSON, nullable=True)


def _normalize_jdbc(url: str) -> str:
    """Convert jdbc:postgresql://host:port/db to postgresql+psycopg2://host:port/db

    Note: This does not add credentials. Prefer passing a full SQLAlchemy URL in DATABASE_URL.
    """
    if not url:
        return url
    if url.startswith('jdbc:postgresql://'):
        return 'postgresql+psycopg2://' + url[len('jdbc:postgresql://'):]
    # if already looks like postgresql URL, ensure psycopg2 driver
    if url.startswith('postgresql://') and 'psycopg2' not in url:
        return url.replace('postgresql://', 'postgresql+psycopg2://', 1)
    return url


def init_db(database_url: Optional[str] = None):
    """Initialize the DB engine and session maker. If database_url is None, tries env DATABASE_URL.

    Returns engine.
    """
    global _engine, _SessionLocal
    if _engine is not None:
        return _engine

    database_url = database_url or os.environ.get('DATABASE_URL') or os.environ.get('JDBC_DATABASE_URL') or os.environ.get('JDBC_URL')
    if not database_url:
        # DB not configured; leave as None
        return None

    database_url = _normalize_jdbc(database_url)
    _engine = create_engine(database_url, echo=False, future=True)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    # create tables if not exist
    create_tables(_engine)
    return _engine


def create_tables(engine):
    Base.metadata.create_all(engine)


def get_session():
    if _SessionLocal is None:
        raise RuntimeError('Database not initialized. Call init_db(database_url) first.')
    return _SessionLocal()


def save_message(role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Persist a chat message and return a dict of the saved row."""
    engine = _engine
    if engine is None:
        raise RuntimeError('Database not initialized.')
    session = get_session()
    try:
        m = ChatMessage(role=role, content=content, message_metadata=metadata)
        session.add(m)
        session.commit()
        session.refresh(m)
        return {"id": m.id, "role": m.role, "content": m.content, "created_at": m.created_at.isoformat(), "metadata": m.message_metadata}
    finally:
        session.close()


def load_recent_messages(limit: int = 100) -> List[Dict[str, Any]]:
    """Load recent chat messages ordered by created_at ascending (oldest first).

    Returns list of dicts.
    """
    engine = _engine
    if engine is None:
        raise RuntimeError('Database not initialized.')
    session = get_session()
    try:
        rows = session.query(ChatMessage).order_by(ChatMessage.created_at.asc()).limit(limit).all()
        return [{"id": r.id, "role": r.role, "content": r.content, "created_at": r.created_at.isoformat(), "metadata": r.message_metadata} for r in rows]
    finally:
        session.close()
