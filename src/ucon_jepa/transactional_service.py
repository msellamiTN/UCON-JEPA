"""G7.3 transactional reference service for joint UCON/CAS enforcement.

This is an executable single-database reference semantics, not a production
distributed transaction coordinator. All protected state changes share one
SQLite write-transaction domain. SQLite BEGIN IMMEDIATE serializes writers;
the authorization check and resource update occur before the same COMMIT.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Callable, Optional


class IdempotencyConflict(RuntimeError):
    """An invocation ID was reused for a different command."""


@dataclass(frozen=True)
class MutationResult:
    invocation_id: str
    outcome: str
    reason: str
    resource_version: Optional[int]
    policy_version: int
    authorization_epoch: int
    event_sequence: int
    receipt_id: Optional[str] = None
    replayed: bool = False


class SqliteUconService:
    """Atomic policy/epoch/resource reference monitor backed by one SQLite DB.

    All methods open independent connections. A caller must place this database
    on storage with SQLite's documented locking semantics. Do not put it on an
    unreliable network filesystem or infer distributed-database guarantees.
    """

    def __init__(
        self, database: str | Path, *, busy_timeout_ms: int = 10_000,
        fault_hook: Callable[[str], None] | None = None,
    ):
        self.database = str(database)
        self.busy_timeout_ms = busy_timeout_ms
        self.fault_hook = fault_hook
        if self.busy_timeout_ms <= 0:
            raise ValueError("busy_timeout_ms must be positive")
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.database, timeout=self.busy_timeout_ms / 1000,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={int(self.busy_timeout_ms)}")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS control_state (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    policy_version INTEGER NOT NULL CHECK(policy_version>=0),
                    authorization_epoch INTEGER NOT NULL CHECK(authorization_epoch>=0)
                );
                INSERT OR IGNORE INTO control_state VALUES (1, 1, 1);

                CREATE TABLE IF NOT EXISTS resources (
                    resource_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL CHECK(version>=0),
                    payload BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS grants (
                    principal TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    PRIMARY KEY(principal, action, resource_id)
                );
                CREATE TABLE IF NOT EXISTS revoked (
                    principal TEXT PRIMARY KEY
                );
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    invocation_id TEXT,
                    principal TEXT,
                    action TEXT,
                    resource_id TEXT,
                    outcome TEXT NOT NULL,
                    policy_version INTEGER NOT NULL,
                    authorization_epoch INTEGER NOT NULL,
                    resource_version INTEGER
                );
                CREATE TABLE IF NOT EXISTS invocations (
                    invocation_id TEXT PRIMARY KEY,
                    command_digest TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    resource_version INTEGER,
                    policy_version INTEGER NOT NULL,
                    authorization_epoch INTEGER NOT NULL,
                    event_sequence INTEGER NOT NULL,
                    receipt_id TEXT
                );
            """)

    @staticmethod
    def _begin(conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")

    @staticmethod
    def _finish(conn: sqlite3.Connection) -> None:
        conn.commit()

    @staticmethod
    def _rollback(conn: sqlite3.Connection) -> None:
        if conn.in_transaction:
            conn.rollback()

    def create_resource(self, resource_id: str, payload: bytes = b"{}") -> None:
        if not resource_id:
            raise ValueError("resource_id is required")
        conn = self._connect()
        try:
            self._begin(conn)
            conn.execute(
                "INSERT INTO resources(resource_id,version,payload) VALUES(?,0,?)",
                (resource_id, sqlite3.Binary(payload)),
            )
            self._finish(conn)
        except Exception:
            self._rollback(conn)
            raise
        finally:
            conn.close()

    def grant(self, principal: str, action: str, resource_id: str) -> tuple[int, int]:
        """Install/update a grant and advance both policy version and auth epoch."""
        if not all((principal, action, resource_id)):
            raise ValueError("grant binding fields are required")
        conn = self._connect()
        try:
            self._begin(conn)
            conn.execute(
                "INSERT OR REPLACE INTO grants VALUES(?,?,?)",
                (principal, action, resource_id),
            )
            conn.execute(
                "UPDATE control_state SET policy_version=policy_version+1, "
                "authorization_epoch=authorization_epoch+1 WHERE singleton=1"
            )
            state = conn.execute(
                "SELECT policy_version,authorization_epoch FROM control_state WHERE singleton=1"
            ).fetchone()
            self._finish(conn)
            return state["policy_version"], state["authorization_epoch"]
        except Exception:
            self._rollback(conn)
            raise
        finally:
            conn.close()

    def revoke(self, principal: str) -> tuple[int, int, int]:
        """Revoke a principal and advance the authorization epoch atomically."""
        if not principal:
            raise ValueError("principal is required")
        conn = self._connect()
        try:
            self._begin(conn)
            conn.execute("INSERT OR IGNORE INTO revoked VALUES(?)", (principal,))
            conn.execute(
                "UPDATE control_state SET authorization_epoch=authorization_epoch+1 "
                "WHERE singleton=1"
            )
            state = conn.execute(
                "SELECT policy_version,authorization_epoch FROM control_state WHERE singleton=1"
            ).fetchone()
            cur = conn.execute(
                """INSERT INTO events(event_type,outcome,policy_version,
                   authorization_epoch) VALUES('REVOKE','COMMITTED',?,?)""",
                (state["policy_version"], state["authorization_epoch"]),
            )
            seq = cur.lastrowid
            self._finish(conn)
            return state["policy_version"], state["authorization_epoch"], seq
        except Exception:
            self._rollback(conn)
            raise
        finally:
            conn.close()

    def update_policy(self) -> tuple[int, int]:
        """Advance both policy version and authorization epoch."""
        conn = self._connect()
        try:
            self._begin(conn)
            conn.execute(
                "UPDATE control_state SET policy_version=policy_version+1, "
                "authorization_epoch=authorization_epoch+1 WHERE singleton=1"
            )
            state = conn.execute(
                "SELECT policy_version,authorization_epoch FROM control_state WHERE singleton=1"
            ).fetchone()
            self._finish(conn)
            return state["policy_version"], state["authorization_epoch"]
        except Exception:
            self._rollback(conn)
            raise
        finally:
            conn.close()

    def snapshot(self, resource_id: str) -> dict:
        conn = self._connect()
        try:
            resource = conn.execute(
                "SELECT version,payload FROM resources WHERE resource_id=?",
                (resource_id,),
            ).fetchone()
            state = conn.execute(
                "SELECT policy_version,authorization_epoch FROM control_state WHERE singleton=1"
            ).fetchone()
            if resource is None:
                raise KeyError(resource_id)
            return {
                "resource_version": resource["version"],
                "payload": bytes(resource["payload"]),
                "policy_version": state["policy_version"],
                "authorization_epoch": state["authorization_epoch"],
            }
        finally:
            conn.close()

    @staticmethod
    def _result(row: sqlite3.Row, *, replayed: bool = False) -> MutationResult:
        return MutationResult(
            invocation_id=row["invocation_id"], outcome=row["outcome"],
            reason=row["reason"], resource_version=row["resource_version"],
            policy_version=row["policy_version"],
            authorization_epoch=row["authorization_epoch"],
            event_sequence=row["event_sequence"], receipt_id=row["receipt_id"],
            replayed=replayed,
        )

    def mutate(
        self, *, invocation_id: str, principal: str, action: str,
        resource_id: str, expected_resource_version: int,
        expected_policy_version: int, expected_authorization_epoch: int,
        payload: bytes,
    ) -> MutationResult:
        """Authorize and mutate within one serializable SQLite writer transaction.

        The command is bound to a unique invocation ID. Identical retries return
        the original result; a key reused for a different command is rejected.
        Rejected attempts are durably recorded, so the same invocation ID cannot
        later be repurposed after state changes.
        """
        if not all((invocation_id, principal, action, resource_id)):
            raise ValueError("invocation and authorization binding fields are required")
        if min(expected_resource_version, expected_policy_version,
               expected_authorization_epoch) < 0:
            raise ValueError("expected versions must be non-negative")
        command_digest = hashlib.sha256(json.dumps({
            "principal": principal, "action": action, "resource_id": resource_id,
            "expected_resource_version": expected_resource_version,
            "expected_policy_version": expected_policy_version,
            "expected_authorization_epoch": expected_authorization_epoch,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

        conn = self._connect()
        try:
            self._begin(conn)
            prior = conn.execute(
                "SELECT * FROM invocations WHERE invocation_id=?", (invocation_id,)
            ).fetchone()
            if prior is not None:
                if prior["command_digest"] != command_digest:
                    raise IdempotencyConflict("invocation ID reused with different command")
                self._finish(conn)
                return self._result(prior, replayed=True)

            state = conn.execute(
                "SELECT policy_version,authorization_epoch FROM control_state WHERE singleton=1"
            ).fetchone()
            resource = conn.execute(
                "SELECT version FROM resources WHERE resource_id=?", (resource_id,)
            ).fetchone()
            current_policy = state["policy_version"]
            current_epoch = state["authorization_epoch"]
            current_resource_version = None if resource is None else resource["version"]

            outcome, reason = "DENIED", "UCON grant absent"
            if resource is None:
                outcome, reason = "REJECTED", "resource not found"
            elif expected_policy_version != current_policy:
                outcome, reason = "CONFLICT", "policy version changed"
            elif expected_authorization_epoch != current_epoch:
                outcome, reason = "CONFLICT", "authorization epoch changed"
            elif expected_resource_version != current_resource_version:
                outcome, reason = "CONFLICT", "resource version changed"
            else:
                is_revoked = conn.execute(
                    "SELECT 1 FROM revoked WHERE principal=?", (principal,)
                ).fetchone()
                has_grant = conn.execute(
                    "SELECT 1 FROM grants WHERE principal=? AND action=? AND resource_id=?",
                    (principal, action, resource_id),
                ).fetchone()
                if is_revoked:
                    outcome, reason = "DENIED", "principal revoked"
                elif has_grant is None:
                    outcome, reason = "DENIED", "UCON grant absent"
                else:
                    outcome, reason = "ACCEPTED", "atomic authorization and CAS succeeded"

            new_version = current_resource_version
            receipt_id = None
            if outcome == "ACCEPTED":
                new_version = current_resource_version + 1
                conn.execute(
                    "UPDATE resources SET version=?,payload=? WHERE resource_id=? AND version=?",
                    (new_version, sqlite3.Binary(payload), resource_id,
                     expected_resource_version),
                )
                if conn.total_changes < 1:
                    raise RuntimeError("resource CAS invariant violated")
                receipt_id = hashlib.sha256(
                    (invocation_id + ":" + command_digest).encode()
                ).hexdigest()

            event = conn.execute(
                """INSERT INTO events(event_type,invocation_id,principal,action,
                   resource_id,outcome,policy_version,authorization_epoch,resource_version)
                   VALUES('MUTATION',?,?,?,?,?,?,?,?)""",
                (invocation_id, principal, action, resource_id, outcome,
                 current_policy, current_epoch, new_version),
            )
            seq = event.lastrowid
            conn.execute(
                """INSERT INTO invocations VALUES(?,?,?,?,?,?,?,?,?)""",
                (invocation_id, command_digest, outcome, reason, new_version,
                 current_policy, current_epoch, seq, receipt_id),
            )
            row = conn.execute(
                "SELECT * FROM invocations WHERE invocation_id=?", (invocation_id,)
            ).fetchone()
            if self.fault_hook is not None:
                self.fault_hook("before_commit")
            self._finish(conn)
            if self.fault_hook is not None:
                self.fault_hook("after_commit")
            return self._result(row)
        except Exception:
            self._rollback(conn)
            raise
        finally:
            conn.close()
