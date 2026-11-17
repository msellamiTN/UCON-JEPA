"""G7.2 reference gateway identity and credential mediation.

Injected interfaces are trust boundaries. Production deployments must provide
cryptographic identity verification, policy evaluation, a scoped secret broker,
and network-enforced gateway-only service access.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Mapping, Protocol
import time


class GatewayDenied(RuntimeError):
    pass


@dataclass(frozen=True)
class VerifiedIdentity:
    issuer: str
    subject: str
    audience: str
    expires_at: int
    not_before: int
    tenant: str
    delegation: str | None = None


class IdentityVerifier(Protocol):
    def verify(self, bearer_token: str) -> VerifiedIdentity:
        """Cryptographically validate token and return trusted claims."""


@dataclass(frozen=True)
class Route:
    service_audience: str
    method: str
    path_template: str
    capability: str


class RouteRegistry(Protocol):
    def resolve(self, operation: str) -> Route: ...


@dataclass(frozen=True)
class PolicyPermit:
    allowed: bool
    effective_principal: str
    policy_version: int
    authorization_epoch: int
    reason: str = ""


class PolicyDecisionPoint(Protocol):
    def authorize(
        self, *, identity: VerifiedIdentity, action: str, resource: str,
        capability: str,
    ) -> PolicyPermit: ...


@dataclass(frozen=True)
class CredentialScope:
    audience: str
    principal: str
    action: str
    resource: str
    ttl_seconds: int


@dataclass(frozen=True)
class ScopedCredential:
    # repr=False prevents accidental disclosure in dataclass logs/debug output.
    value: str = field(repr=False)
    expires_at: int


class CredentialBroker(Protocol):
    def mint(self, scope: CredentialScope) -> ScopedCredential: ...


@dataclass(frozen=True)
class InboundRequest:
    bearer_token: str = field(repr=False)
    operation: str
    resource_key: str
    body: bytes
    # Deliberately no target URL, effective principal, or downstream credential.


@dataclass(frozen=True)
class DownstreamRequest:
    method: str
    target: str
    headers: Mapping[str, str] = field(repr=False)
    body: bytes = field(repr=False)


@dataclass(frozen=True)
class GatewayReceipt:
    status: int
    invocation_id: str
    principal: str
    target: str
    policy_version: int
    authorization_epoch: int
    body_sha256: str


class DownstreamTransport(Protocol):
    def send(self, request: DownstreamRequest) -> int: ...


class GatewayMediator:
    def __init__(
        self, *, verifier: IdentityVerifier, routes: RouteRegistry,
        policy: PolicyDecisionPoint, broker: CredentialBroker,
        transport: DownstreamTransport, expected_issuer: str,
        accepted_agent_audience: str, now=None, credential_ttl: int = 60,
    ):
        if credential_ttl <= 0:
            raise ValueError("credential_ttl must be positive")
        self.verifier = verifier
        self.routes = routes
        self.policy = policy
        self.broker = broker
        self.transport = transport
        self.expected_issuer = expected_issuer
        self.accepted_agent_audience = accepted_agent_audience
        self.now = now or (lambda: int(time.time()))
        self.credential_ttl = credential_ttl
        self._invocations: set[str] = set()

    def execute(self, request: InboundRequest, *, invocation_id: str) -> GatewayReceipt:
        if not invocation_id or invocation_id in self._invocations:
            raise GatewayDenied("missing or duplicate invocation id")
        if not request.bearer_token or not request.operation or not request.resource_key:
            raise GatewayDenied("missing authentication or operation binding")

        # Only the verifier may turn untrusted token bytes into trusted claims.
        identity = self.verifier.verify(request.bearer_token)
        now = int(self.now())
        if identity.issuer != self.expected_issuer:
            raise GatewayDenied("untrusted issuer")
        if identity.audience != self.accepted_agent_audience:
            raise GatewayDenied("wrong agent audience")
        if not identity.subject or not identity.tenant:
            raise GatewayDenied("incomplete verified identity")
        if identity.expires_at <= now or identity.not_before > now:
            raise GatewayDenied("identity token outside validity interval")

        route = self.routes.resolve(request.operation)
        if not route.service_audience or not route.path_template.startswith("/"):
            raise GatewayDenied("invalid trusted route configuration")
        # Resource key is a path segment, not an arbitrary URL/path expression.
        if "/" in request.resource_key or request.resource_key in (".", ".."):
            raise GatewayDenied("invalid resource key")
        target = route.path_template.replace("{resource}", request.resource_key)
        if "{resource}" in target or target.startswith("//") or "://" in target:
            raise GatewayDenied("route template is not a fixed local path")

        permit = self.policy.authorize(
            identity=identity, action=route.method, resource=target,
            capability=route.capability,
        )
        if not permit.allowed:
            raise GatewayDenied("UCON policy denied: " + permit.reason)
        if not permit.effective_principal:
            raise GatewayDenied("policy did not derive an effective principal")

        scope = CredentialScope(
            audience=route.service_audience,
            principal=permit.effective_principal,
            action=route.method,
            resource=target,
            ttl_seconds=self.credential_ttl,
        )
        credential = self.broker.mint(scope)
        if not credential.value or credential.expires_at <= now:
            raise GatewayDenied("broker returned invalid credential")
        if credential.expires_at > now + self.credential_ttl:
            raise GatewayDenied("broker credential exceeds requested TTL")

        body_digest = sha256(request.body).hexdigest()
        # Construct a fresh allowlisted header map; never forward inbound headers.
        downstream = DownstreamRequest(
            method=route.method,
            target=target,
            headers={
                "Authorization": "Bearer " + credential.value,
                "X-Effective-Principal": permit.effective_principal,
                "X-UCON-Invocation-ID": invocation_id,
                "X-UCON-Policy-Version": str(permit.policy_version),
                "X-UCON-Authorization-Epoch": str(permit.authorization_epoch),
                "X-UCON-Capability": route.capability,
                "X-Content-SHA256": body_digest,
            },
            body=request.body,
        )
        self._invocations.add(invocation_id)
        status = self.transport.send(downstream)
        return GatewayReceipt(
            status=status, invocation_id=invocation_id,
            principal=permit.effective_principal, target=target,
            policy_version=permit.policy_version,
            authorization_epoch=permit.authorization_epoch,
            body_sha256=body_digest,
        )
