"""G7 synchronous conditional HTTP mutation adapter.

Research reference code. It does not implement identity, cryptographic signing,
durable recovery, or a production authorization gateway.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Mapping, Protocol


class Outcome(str, Enum):
    ACCEPTED = "ACCEPTED"
    CONFLICT = "CONFLICT"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class GateDecision:
    invocation_id: str
    effective_principal: str
    method: str
    target: str
    capability: str
    policy_version: int
    authorization_epoch: int
    resource_etag: str
    allowed: bool


@dataclass(frozen=True)
class Mutation:
    body: bytes
    content_type: str = "application/json"


@dataclass(frozen=True)
class HttpRequest:
    method: str
    target: str
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes = b""


@dataclass(frozen=True)
class AdapterResult:
    outcome: Outcome
    invocation_id: str
    receipt_id: str | None = None
    response_status: int | None = None
    body_sha256: str | None = None
    reason: str = ""


class Transport(Protocol):
    def send(self, request: HttpRequest) -> HttpResponse:
        """Send exactly once. Implementations must not retry internally."""


class TransportFailure(Exception):
    """The request may or may not have reached the remote service."""


class ConditionalAdapter:
    """One-shot adapter; success is acceptance, never verification."""

    SAFE_METHODS = frozenset({"PUT", "PATCH", "POST", "DELETE"})

    def __init__(self, transport: Transport):
        self._transport = transport
        self._used_invocations: set[str] = set()

    @staticmethod
    def _valid_strong_etag(etag: str) -> bool:
        return bool(etag) and not etag.startswith("W/") and etag.startswith('"') and etag.endswith('"')

    def mutate(self, decision: GateDecision, mutation: Mutation) -> AdapterResult:
        # Validate the immutable gate result before any network side effect.
        if not decision.allowed:
            return AdapterResult(Outcome.REJECTED, decision.invocation_id, reason="gate denied")
        required = (
            decision.invocation_id, decision.effective_principal, decision.target,
            decision.capability,
        )
        if not all(required):
            return AdapterResult(Outcome.REJECTED, decision.invocation_id, reason="incomplete binding")
        method = decision.method.upper()
        if method not in self.SAFE_METHODS:
            return AdapterResult(Outcome.REJECTED, decision.invocation_id, reason="unsupported mutation method")
        if not self._valid_strong_etag(decision.resource_etag):
            return AdapterResult(Outcome.REJECTED, decision.invocation_id, reason="strong ETag required")
        if decision.policy_version < 0 or decision.authorization_epoch < 0:
            return AdapterResult(Outcome.REJECTED, decision.invocation_id, reason="invalid policy/authorization epoch")
        if decision.invocation_id in self._used_invocations:
            return AdapterResult(Outcome.REJECTED, decision.invocation_id, reason="duplicate invocation")
        self._used_invocations.add(decision.invocation_id)

        digest = sha256(mutation.body).hexdigest()
        request = HttpRequest(
            method=method,
            target=decision.target,
            headers={
                "Content-Type": mutation.content_type,
                "If-Match": decision.resource_etag,
                "Idempotency-Key": decision.invocation_id,
                # This header is meaningful only if the server atomically checks it
                # with the resource mutation; a generic API may ignore it.
                "X-UCON-Policy-Version": str(decision.policy_version),
                "X-UCON-Authorization-Epoch": str(decision.authorization_epoch),
                "X-UCON-Effective-Principal": decision.effective_principal,
                "X-UCON-Capability": decision.capability,
                "X-UCON-Body-SHA256": digest,
            },
            body=mutation.body,
        )
        try:
            response = self._transport.send(request)
        except (TransportFailure, TimeoutError, ConnectionError) as exc:
            return AdapterResult(
                Outcome.UNKNOWN, decision.invocation_id, body_sha256=digest,
                reason=f"transport outcome ambiguous: {type(exc).__name__}",
            )

        if response.status == 412:
            return AdapterResult(Outcome.CONFLICT, decision.invocation_id,
                                 response_status=412, body_sha256=digest,
                                 reason="conditional precondition failed")
        if response.status in (401, 403):
            return AdapterResult(Outcome.REJECTED, decision.invocation_id,
                                 response_status=response.status, body_sha256=digest,
                                 reason="remote service rejected request")
        if response.status in (200, 201, 204):
            receipt = response.headers.get("X-Receipt-ID")
            if not receipt:
                return AdapterResult(Outcome.UNKNOWN, decision.invocation_id,
                                     response_status=response.status, body_sha256=digest,
                                     reason="success response lacks correlated receipt")
            return AdapterResult(Outcome.ACCEPTED, decision.invocation_id,
                                 receipt_id=receipt, response_status=response.status,
                                 body_sha256=digest, reason="remote acceptance only; verification required")
        return AdapterResult(Outcome.UNKNOWN, decision.invocation_id,
                             response_status=response.status, body_sha256=digest,
                             reason="status does not prove absence or presence of mutation")
