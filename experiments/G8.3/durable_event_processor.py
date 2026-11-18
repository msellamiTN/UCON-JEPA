#!/usr/bin/env python3
"""Transactional SQLite replay ledger + signed UCON trace processor.

SQLite BEGIN IMMEDIATE serializes writers against one local database. This is
not a distributed consensus/ledger service and must not be placed on an
unvalidated shared network filesystem.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from authenticated_evaluation import evaluate_signed_trace


class LedgerError(RuntimeError):
    """Raised when the transactional event-processing boundary fails."""


class SQLiteEventProcessor:
    def __init__(self, database: str | Path):
        self.database = str(database)
        with sqlite3.connect(self.database) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS processed_events (
                    event_id TEXT PRIMARY KEY,
                    issuer TEXT NOT NULL,
                    key_id TEXT NOT NULL,
                    envelope_json TEXT NOT NULL,
                    accepted_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    accepted_at INTEGER NOT NULL
                );
            """)

    def process(
        self,
        signed_events: Sequence[Mapping[str, Any]],
        trusted_keys: Mapping[str, Mapping[str, Any]],
        trusted_entitlements: Mapping[str, list[str]] | None = None,
    ) -> dict:
        """Authenticate, evaluate, and commit replay IDs + decision atomically."""
        connection = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            seen = {row[0] for row in connection.execute(
                "SELECT event_id FROM processed_events"
            )}
            result = evaluate_signed_trace(
                signed_events, trusted_keys, trusted_entitlements,
                seen_event_ids=seen,
            )
            accepted_at = int(time.time())
            evaluation_id = str(uuid.uuid4())
            for envelope in signed_events:
                connection.execute(
                    """INSERT INTO processed_events
                       (event_id, issuer, key_id, envelope_json, accepted_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        envelope["event_id"], envelope["issuer"], envelope["key_id"],
                        json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                        accepted_at,
                    ),
                )
            connection.execute(
                "INSERT INTO evaluations (evaluation_id, result_json, accepted_at) VALUES (?, ?, ?)",
                (evaluation_id, json.dumps(result, sort_keys=True), accepted_at),
            )
            connection.execute("COMMIT")
            return {"evaluation_id": evaluation_id, **result, "committed": True}
        except Exception as exc:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise exc
        finally:
            connection.close()

    def counts(self) -> dict:
        with sqlite3.connect(self.database) as connection:
            events = connection.execute("SELECT COUNT(*) FROM processed_events").fetchone()[0]
            evaluations = connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
        return {"processed_events": events, "evaluations": evaluations}
