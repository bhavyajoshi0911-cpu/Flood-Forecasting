"""SQLite engine and schema initialization helpers."""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models.database_models import Base


def get_engine(database_url: str | None = None) -> Engine:
    """Create an engine without creating tables as an import side effect."""

    url = database_url or os.getenv("DATABASE_URL", "sqlite:///./flood_forecasting.db")
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=engine or get_engine(), autoflush=False, expire_on_commit=False)


def init_database(engine: Engine | str | None = None) -> Engine:
    """Create all declared tables and return the engine used."""

    active_engine = get_engine(engine) if isinstance(engine, str) else engine or get_engine()
    Base.metadata.create_all(active_engine)
    return active_engine