"""PostgreSQL refinement of the G7 atomic UCON/CAS reference service.

Requires psycopg 3. All authorization and resource state must reside in this
database. The singleton control row is locked FOR UPDATE by every protected
writer, establishing one serialization order for grant/revoke/policy/mutate.
This is a reference adapter, not a claim about arbitrary external effects.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .transactional_service import IdempotencyConflict, MutationResult


@dataclass(frozen=True)
class PostgresConfig:
    dsn: str
    connect_timeout: int = 5


class PostgresUconService:
    def __init__(
        self, config: PostgresConfig, *,
        fault_hook: Callable[[str], None] | None = None,
    ):
        if not config.dsn:
            raise ValueError("PostgreSQL DSN is required")
        self.config = config
        self.fault_hook = fault_hook

    def _connect(self):
        return psycopg.connect(
            self.config.dsn, connect_timeout=self.config.connect_timeout,
            row_factory=dict_row,
        )

    def initialize(self) -> None:
        """Apply idempotent reference schema DDL; production should use migrations."""
        ddl = """
        CREATE TABLE IF NOT EXISTS ucon_control_state (
            singleton SMALLINT PRIMARY KEY CHECK (singleton = 1),
            policy_version BIGINT NOT NULL CHECK (policy_version >= 0),
            authorization_epoch BIGINT NOT NULL CHECK (authorization_epoch >= 0)
        );
        INSERT INTO ucon_control_state VALUES (1, 1, 1)
            ON CONFLICT (singleton) DO NOTHING;
        CREATE TABLE IF NOT EXISTS ucon_resources (
            resource_id TEXT PRIMARY KEY,
            version BIGINT NOT NULL CHECK (version >= 0),
            payload BYTEA NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ucon_grants (
            principal TEXT NOT NULL, action TEXT NOT NULL, resource_id TEXT NOT NULL,
            PRIMARY KEY (principal, action, resource_id)
        );
        CREATE TABLE IF NOT EXISTS ucon_revoked (
            principal TEXT PRIMARY KEY
        );
        CREATE TABLE IF NOT EXISTS ucon_events (
            sequence BIGSERIAL PRIMARY KEY, event_type TEXT NOT NULL,
            invocation_id TEXT, principal TEXT, action TEXT, resource_id TEXT,
            outcome TEXT NOT NULL, policy_version BIGINT NOT NULL,
            authorization_epoch BIGINT NOT NULL, resource_version BIGINT
        );
        CREATE TABLE IF NOT EXISTS ucon_invocations (
            invocation_id TEXT PRIMARY KEY, command_digest TEXT NOT NULL,
            outcome TEXT NOT NULL, reason TEXT NOT NULL,
            resource_version BIGINT, policy_version BIGINT NOT NULL,
            authorization_epoch BIGINT NOT NULL, event_sequence BIGINT NOT NULL,
            receipt_id TEXT
        );
        CREATE TABLE IF NOT EXISTS ucon_outbox (
            event_id TEXT PRIMARY KEY,
            invocation_id TEXT NOT NULL UNIQUE,
            event_type TEXT NOT NULL,
            payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
            available_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            lease_until TIMESTAMPTZ,
            delivered_at TIMESTAMPTZ
        );
        CREATE INDEX IF NOT EXISTS ucon_outbox_ready_idx
            ON ucon_outbox(available_at, created_at)
            WHERE delivered_at IS NULL;
        """
        with self._connect() as conn:
            for statement in ddl.split(";"):
                if statement.strip():
                    conn.execute(statement)

    @staticmethod
    def _lock_control(cur):
        row = cur.execute(
            "SELECT policy_version, authorization_epoch FROM ucon_control_state "
            "WHERE singleton=1 FOR UPDATE"
        ).fetchone()
        if row is None:
            raise RuntimeError("ucon_control_state is not initialized")
        return row

    def create_resource(self, resource_id: str, payload: bytes = b"{}") -> None:
        if not resource_id:
            raise ValueError("resource_id is required")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO ucon_resources(resource_id,version,payload) VALUES(%s,0,%s)",
                (resource_id, payload),
            )

    def grant(self, principal: str, action: str, resource_id: str) -> tuple[int, int]:
        if not all((principal, action, resource_id)):
            raise ValueError("grant binding fields are required")
        with self._connect() as conn:
            with conn.cursor() as cur:
                self._lock_control(cur)
                cur.execute(
                    "INSERT INTO ucon_grants VALUES(%s,%s,%s) "
                    "ON CONFLICT(principal,action,resource_id) DO NOTHING",
                    (principal, action, resource_id),
                )
                state = cur.execute(
                    "UPDATE ucon_control_state SET policy_version=policy_version+1, "
                    "authorization_epoch=authorization_epoch+1 WHERE singleton=1 "
                    "RETURNING policy_version,authorization_epoch"
                ).fetchone()
                return state["policy_version"], state["authorization_epoch"]

    def revoke(self, principal: str) -> tuple[int, int, int]:
        if not principal:
            raise ValueError("principal is required")
        with self._connect() as conn:
            with conn.cursor() as cur:
                self._lock_control(cur)
                cur.execute(
                    "INSERT INTO ucon_revoked VALUES(%s) ON CONFLICT(principal) DO NOTHING",
                    (principal,),
                )
                state = cur.execute(
                    "UPDATE ucon_control_state SET authorization_epoch=authorization_epoch+1 "
                    "WHERE singleton=1 RETURNING policy_version,authorization_epoch"
                ).fetchone()
                seq = cur.execute(
                    "INSERT INTO ucon_events(event_type,outcome,policy_version,"
                    "authorization_epoch) VALUES('REVOKE','COMMITTED',%s,%s) "
                    "RETURNING sequence",
                    (state["policy_version"], state["authorization_epoch"]),
                ).fetchone()["sequence"]
                return state["policy_version"], state["authorization_epoch"], seq

    def update_policy(self) -> tuple[int, int]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                self._lock_control(cur)
                state = cur.execute(
                    "UPDATE ucon_control_state SET policy_version=policy_version+1, "
                    "authorization_epoch=authorization_epoch+1 WHERE singleton=1 "
                    "RETURNING policy_version,authorization_epoch"
                ).fetchone()
                return state["policy_version"], state["authorization_epoch"]

    def snapshot(self, resource_id: str) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT r.version,r.payload,c.policy_version,c.authorization_epoch "
                "FROM ucon_resources r CROSS JOIN ucon_control_state c "
                "WHERE r.resource_id=%s AND c.singleton=1",
                (resource_id,),
            ).fetchone()
            if row is None:
                raise KeyError(resource_id)
            return {
                "resource_version": row["version"], "payload": bytes(row["payload"]),
                "policy_version": row["policy_version"],
                "authorization_epoch": row["authorization_epoch"],
            }

    def mutate(
        self, *, invocation_id: str, principal: str, action: str,
        resource_id: str, expected_resource_version: int,
        expected_policy_version: int, expected_authorization_epoch: int,
        payload: bytes,
    ) -> MutationResult:
        if not all((invocation_id, principal, action, resource_id)):
            raise ValueError("invocation and authorization binding fields are required")
        if min(expected_resource_version, expected_policy_version,
               expected_authorization_epoch) < 0:
            raise ValueError("expected versions must be non-negative")
        digest = hashlib.sha256(json.dumps({
            "principal": principal, "action": action, "resource_id": resource_id,
            "expected_resource_version": expected_resource_version,
            "expected_policy_version": expected_policy_version,
            "expected_authorization_epoch": expected_authorization_epoch,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

        with self._connect() as conn:
            with conn.cursor() as cur:
                state = self._lock_control(cur)
                prior = cur.execute(
                    "SELECT * FROM ucon_invocations WHERE invocation_id=%s",
                    (invocation_id,),
                ).fetchone()
                if prior is not None:
                    if prior["command_digest"] != digest:
                        raise IdempotencyConflict(
                            "invocation ID reused with different command"
                        )
                    return MutationResult(
                        invocation_id=prior["invocation_id"], outcome=prior["outcome"],
                        reason=prior["reason"], resource_version=prior["resource_version"],
                        policy_version=prior["policy_version"],
                        authorization_epoch=prior["authorization_epoch"],
                        event_sequence=prior["event_sequence"],
                        receipt_id=prior["receipt_id"], replayed=True,
                    )

                resource = cur.execute(
                    "SELECT version FROM ucon_resources WHERE resource_id=%s FOR UPDATE",
                    (resource_id,),
                ).fetchone()
                rv = None if resource is None else resource["version"]
                pv, epoch = state["policy_version"], state["authorization_epoch"]
                outcome, reason = "DENIED", "UCON grant absent"
                if resource is None:
                    outcome, reason = "REJECTED", "resource not found"
                elif expected_policy_version != pv:
                    outcome, reason = "CONFLICT", "policy version changed"
                elif expected_authorization_epoch != epoch:
                    outcome, reason = "CONFLICT", "authorization epoch changed"
                elif expected_resource_version != rv:
                    outcome, reason = "CONFLICT", "resource version changed"
                else:
                    revoked = cur.execute(
                        "SELECT 1 FROM ucon_revoked WHERE principal=%s", (principal,)
                    ).fetchone()
                    grant = cur.execute(
                        "SELECT 1 FROM ucon_grants WHERE principal=%s AND action=%s "
                        "AND resource_id=%s", (principal, action, resource_id)
                    ).fetchone()
                    if revoked:
                        outcome, reason = "DENIED", "principal revoked"
                    elif grant:
                        outcome, reason = "ACCEPTED", "atomic authorization and CAS succeeded"

                new_version, receipt = rv, None
                if outcome == "ACCEPTED":
                    new_version = rv + 1
                    updated = cur.execute(
                        "UPDATE ucon_resources SET version=%s,payload=%s "
                        "WHERE resource_id=%s AND version=%s RETURNING version",
                        (new_version, payload, resource_id, expected_resource_version),
                    ).fetchone()
                    if updated is None:
                        raise RuntimeError("resource CAS invariant violated")
                    receipt = hashlib.sha256(
                        (invocation_id + ":" + digest).encode()
                    ).hexdigest()

                seq = cur.execute(
                    "INSERT INTO ucon_events(event_type,invocation_id,principal,action,"
                    "resource_id,outcome,policy_version,authorization_epoch,resource_version) "
                    "VALUES('MUTATION',%s,%s,%s,%s,%s,%s,%s,%s) RETURNING sequence",
                    (invocation_id, principal, action, resource_id, outcome,
                     pv, epoch, new_version),
                ).fetchone()["sequence"]
                cur.execute(
                    "INSERT INTO ucon_invocations VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (invocation_id, digest, outcome, reason, new_version,
                     pv, epoch, seq, receipt),
                )
                if outcome == "ACCEPTED":
                    cur.execute(
                        "INSERT INTO ucon_outbox(event_id,invocation_id,event_type,payload) "
                        "VALUES(%s,%s,'RESOURCE_MUTATED',%s)",
                        (invocation_id, invocation_id, Jsonb({
                            "invocation_id": invocation_id, "principal": principal,
                            "action": action, "resource_id": resource_id,
                            "resource_version": new_version, "receipt_id": receipt,
                            "policy_version": pv, "authorization_epoch": epoch,
                            "event_sequence": seq,
                        })),
                    )
                if self.fault_hook:
                    self.fault_hook("before_commit")
            conn.commit()
            if self.fault_hook:
                self.fault_hook("after_commit")
            return MutationResult(
                invocation_id, outcome, reason, new_version, pv, epoch, seq, receipt, False
            )
