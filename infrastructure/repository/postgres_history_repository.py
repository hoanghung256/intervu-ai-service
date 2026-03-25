import re
import asyncio
from typing import Optional, List, Dict
from sqlalchemy import DateTime, String, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from .history_repository import HistoryRepository

class Base(DeclarativeBase):
    pass

class PostgresHistoryRepository(HistoryRepository):
    def __init__(self, postgres_dsn: str, table_name: str) -> None:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$", table_name):
            raise ValueError("Invalid table name format.")

        schema_name = None
        simple_table_name = table_name
        if "." in table_name:
            schema_name, simple_table_name = table_name.split(".", 1)

        table_args = {"schema": schema_name} if schema_name else {}

        class UserSession(Base):
            __tablename__ = simple_table_name
            __table_args__ = table_args

            id: Mapped[str] = mapped_column(String, primary_key=True)
            history: Mapped[list[dict]] = mapped_column(
                JSONB,
                nullable=False,
                server_default=text("'[]'::jsonb"),
                default=list,
            )
            last_cv: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
            last_cv_pdf_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
            transcript: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
            updated_at: Mapped[object] = mapped_column(
                DateTime(timezone=True),
                nullable=False,
                server_default=func.now(),
                onupdate=func.now(),
            )

        async_dsn = self._to_async_dsn(postgres_dsn)
        self.engine = create_async_engine(async_dsn, pool_pre_ping=True)
        self.SessionLocal = async_sessionmaker(bind=self.engine, autoflush=False, expire_on_commit=False)
        self.UserSession = UserSession
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @staticmethod
    def _to_async_dsn(dsn: str) -> str:
        if dsn.startswith("postgresql+"):
            return dsn
        if dsn.startswith("postgresql:"):
            return re.sub(r"^postgresql:", "postgresql+psycopg:", dsn)
        return dsn

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self._initialized = True

    async def get_history(self, user_id: str) -> List[Dict]:
        await self._ensure_initialized()
        async with self.SessionLocal() as db:
            row = await db.get(self.UserSession, user_id)
            return row.history if row else []

    async def append_message(self, user_id: str, message: Dict, max_messages: Optional[int] = None) -> None:
        await self._ensure_initialized()
        async with self.SessionLocal() as db:
            row = await db.get(self.UserSession, user_id)
            if not row:
                row = self.UserSession(id=user_id, history=[message])
                db.add(row)
            else:
                history = list(row.history or [])
                history.append(message)
                row.history = history
            await db.commit()

    async def clear_history(self, user_id: str) -> None:
        await self._ensure_initialized()
        async with self.SessionLocal() as db:
            row = await db.get(self.UserSession, user_id)
            if row:
                row.history = []
                await db.commit()

    async def get_last_cv(self, user_id: str) -> Optional[Dict]:
        await self._ensure_initialized()
        async with self.SessionLocal() as db:
            row = await db.get(self.UserSession, user_id)
            return row.last_cv if row else None

    async def set_last_cv(self, user_id: str, cv_json: Optional[Dict]) -> None:
        await self._ensure_initialized()
        async with self.SessionLocal() as db:
            row = await db.get(self.UserSession, user_id)
            if not row:
                row = self.UserSession(id=user_id, last_cv=cv_json)
                db.add(row)
            else:
                row.last_cv = cv_json
            await db.commit()

    async def get_last_cv_pdf_url(self, user_id: str) -> Optional[str]:
        await self._ensure_initialized()
        async with self.SessionLocal() as db:
            row = await db.get(self.UserSession, user_id)
            return row.last_cv_pdf_url if row else None

    async def set_last_cv_pdf_url(self, user_id: str, cv_pdf_url: Optional[str]) -> None:
        await self._ensure_initialized()
        async with self.SessionLocal() as db:
            row = await db.get(self.UserSession, user_id)
            if not row:
                row = self.UserSession(id=user_id, last_cv_pdf_url=cv_pdf_url)
                db.add(row)
            else:
                row.last_cv_pdf_url = cv_pdf_url
            await db.commit()

    async def get_transcript(self, user_id: str) -> Optional[Dict]:
        await self._ensure_initialized()
        async with self.SessionLocal() as db:
            row = await db.get(self.UserSession, user_id)
            return row.transcript if row else None

    async def set_transcript(self, user_id: str, transcript_data: Dict) -> None:
        await self._ensure_initialized()
        async with self.SessionLocal() as db:
            row = await db.get(self.UserSession, user_id)
            if not row:
                row = self.UserSession(id=user_id, transcript=transcript_data)
                db.add(row)
            else:
                row.transcript = transcript_data
            await db.commit()
