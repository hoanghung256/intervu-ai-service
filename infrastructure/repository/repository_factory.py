import os
import logging
from .history_repository import HistoryRepository
from .memory_history_repository import MemoryHistoryRepository
from .postgres_history_repository import PostgresHistoryRepository
from infrastructure.env_constants import (
    DEFAULT_BACKEND,
    DEFAULT_POSTGRES_HISTORY_TABLE,
    ENV_BACKEND,
    ENV_DATABASE_URL,
    ENV_POSTGRES_DSN,
    ENV_POSTGRES_HISTORY_TABLE,
)

_history_repository_instance: HistoryRepository = None

def get_history_repository() -> HistoryRepository:
    global _history_repository_instance
    if _history_repository_instance is not None:
        return _history_repository_instance

    backend = os.getenv(ENV_BACKEND, DEFAULT_BACKEND).lower()

    if backend in {"postgres", "postgresql"}:
        postgres_dsn = os.getenv(ENV_POSTGRES_DSN) or os.getenv(ENV_DATABASE_URL)
        postgres_table = os.getenv(ENV_POSTGRES_HISTORY_TABLE, DEFAULT_POSTGRES_HISTORY_TABLE)
        
        if not postgres_dsn:
            logging.error(f"{ENV_POSTGRES_DSN} not found. Falling back to memory.")
            _history_repository_instance = MemoryHistoryRepository()
        else:
            try:
                _history_repository_instance = PostgresHistoryRepository(
                    postgres_dsn=postgres_dsn, 
                    table_name=postgres_table
                )
                logging.info(f"Initialized PostgreSQL repository: {postgres_table}")
            except Exception as e:
                logging.error(f"Failed to initialize PostgreSQL store: {e}. Falling back to memory.")
                _history_repository_instance = MemoryHistoryRepository()
    else:
        _history_repository_instance = MemoryHistoryRepository()
        logging.info("Initialized Memory repository")

    return _history_repository_instance
