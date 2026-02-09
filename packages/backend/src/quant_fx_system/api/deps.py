"""Dependency providers for the API."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from uuid import uuid4

from fastapi import Depends

from quant_fx_system.settings import Settings, get_settings


def _serialize_for_json(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, default=_serialize_for_json, ensure_ascii=False)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SQLiteStorage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS state_snapshots (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS backtests (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    request TEXT NOT NULL,
                    result TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def save_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        record_id = uuid4().hex
        created_at = _utc_now().isoformat()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO state_snapshots (id, created_at, payload) VALUES (?, ?, ?)",
                (record_id, created_at, _json_dumps(payload)),
            )
            connection.commit()
        return {"id": record_id, "created_at": created_at, "state": payload}

    def latest_state(self) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, created_at, payload FROM state_snapshots ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "state": json.loads(row["payload"]),
        }

    def list_states(self, limit: int) -> Iterable[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, created_at, payload FROM state_snapshots ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "state": json.loads(row["payload"]),
            }
            for row in rows
        ]

    def save_backtest(self, request: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        record_id = uuid4().hex
        created_at = _utc_now().isoformat()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO backtests (id, created_at, request, result) VALUES (?, ?, ?, ?)",
                (record_id, created_at, _json_dumps(request), _json_dumps(result)),
            )
            connection.commit()
        return {
            "id": record_id,
            "created_at": created_at,
            "request": request,
            "result": result,
        }

    def list_backtests(self, limit: int) -> Iterable[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, created_at, request, result FROM backtests ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "request": json.loads(row["request"]),
                "result": json.loads(row["result"]),
            }
            for row in rows
        ]

    def get_backtest(self, record_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, created_at, request, result FROM backtests WHERE id = ?",
                (record_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "request": json.loads(row["request"]),
            "result": json.loads(row["result"]),
        }


@lru_cache(maxsize=1)
def _get_storage(settings: Settings) -> SQLiteStorage:
    return SQLiteStorage(settings.database_path)


def get_storage(settings: Settings = Depends(get_settings)) -> SQLiteStorage:
    return _get_storage(settings)
