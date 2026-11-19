import unittest

from ucon_jepa.gateway_identity import (
    CredentialScope, GatewayDenied, GatewayMediator, InboundRequest,
    PolicyPermit, Route, ScopedCredential, VerifiedIdentity,
)


class FakeVerifier:
    def __init__(self, identity=None, error=None):
        self.identity, self.error, self.calls = identity, error, []

    def verify(self, token):
        self.calls.append(token)
        if self.error:
            raise self.error
        return self.identity


class FakeRoutes:
    def resolve(self, operation):
        if operation != "update-resource":
            raise GatewayDenied("unknown operation")
        return Route("resource-api", "PATCH", "/resources/{resource}", "resource:update")


class FakePolicy:
    def __init__(self, permit=None):
        self.permit = permit or PolicyPermit(True, "tenant-a:agent-7", 9, 12)

    def authorize(self, **kwargs):
        self.last = kwargs
        return self.permit


class FakeBroker:
    def __init__(self):
        self.scopes = []

    def mint(self, scope):
        self.scopes.append(scope)
        return ScopedCredential("scoped-secret", 1040)


class FakeTransport:
    def __init__(self, status=204):
        self.status, self.requests = status, []

    def send(self, request):
        self.requests.append(request)
        return self.status


def make(identity=None, permit=None, now=1000):
    identity = identity or VerifiedIdentity(
        "https://issuer.example", "agent-7", "ucon-agent", 1100, 900, "tenant-a"
    )
    verifier, policy, broker, transport = FakeVerifier(identity), FakePolicy(permit), FakeBroker(), FakeTransport()
    gateway = GatewayMediator(
        verifier=verifier, routes=FakeRoutes(), policy=policy, broker=broker,
        transport=transport, expected_issuer="https://issuer.example",
        accepted_agent_audience="ucon-agent", now=lambda: now, credential_ttl=60,
    )
    return gateway, verifier, policy, broker, transport


class GatewayIdentityTests(unittest.TestCase):
    def request(self, token="signed-token", operation="update-resource", key="42"):
        return InboundRequest(token, operation, key, b'{"status":"ok"}')

    def test_happy_path_derives_principal_and_scopes_credential(self):
        gateway, verifier, policy, broker, transport = make()
        receipt = gateway.execute(self.request(), invocation_id="i-1")
        self.assertEqual(receipt.principal, "tenant-a:agent-7")
        self.assertEqual(broker.scopes[0], CredentialScope(
            "resource-api", "tenant-a:agent-7", "PATCH", "/resources/42", 60
        ))
        self.assertEqual(transport.requests[0].target, "/resources/42")
        self.assertEqual(transport.requests[0].headers["Authorization"], "Bearer scoped-secret")

    def test_wrong_issuer_denied_before_policy_or_network(self):
        ident = VerifiedIdentity("evil", "s", "ucon-agent", 1100, 900, "t")
        gateway, _, policy, broker, transport = make(ident)
        with self.assertRaises(GatewayDenied):
            gateway.execute(self.request(), invocation_id="i")
        self.assertFalse(hasattr(policy, "last"))
        self.assertEqual(broker.scopes, [])
        self.assertEqual(transport.requests, [])

    def test_wrong_audience_denied(self):
        ident = VerifiedIdentity("https://issuer.example", "s", "other", 1100, 900, "t")
        gateway, _, _, broker, transport = make(ident)
        with self.assertRaises(GatewayDenied):
            gateway.execute(self.request(), invocation_id="i")
        self.assertEqual(broker.scopes, [])
        self.assertEqual(transport.requests, [])

    def test_expired_and_not_yet_valid_tokens_denied(self):
        for exp, nbf in ((999, 900), (1200, 1001)):
            ident = VerifiedIdentity("https://issuer.example", "s", "ucon-agent", exp, nbf, "t")
            gateway, _, _, broker, transport = make(ident)
            with self.assertRaises(GatewayDenied):
                gateway.execute(self.request(), invocation_id=f"{exp}-{nbf}")
            self.assertEqual(broker.scopes, [])
            self.assertEqual(transport.requests, [])

    def test_empty_subject_denied(self):
        ident = VerifiedIdentity("https://issuer.example", "", "ucon-agent", 1100, 900, "t")
        gateway, _, _, _, transport = make(ident)
        with self.assertRaises(GatewayDenied):
            gateway.execute(self.request(), invocation_id="i")
        self.assertEqual(transport.requests, [])

    def test_policy_denial_mints_no_credential_and_sends_nothing(self):
        gateway, _, _, broker, transport = make(permit=PolicyPermit(False, "", 9, 12, "no"))
        with self.assertRaises(GatewayDenied):
            gateway.execute(self.request(), invocation_id="i")
        self.assertEqual(broker.scopes, [])
        self.assertEqual(transport.requests, [])

    def test_caller_cannot_choose_target_or_principal(self):
        # The inbound request type has no target/principal fields.
        self.assertNotIn("target", InboundRequest.__dataclass_fields__)
        self.assertNotIn("effective_principal", InboundRequest.__dataclass_fields__)
        gateway, _, _, _, transport = make()
        gateway.execute(self.request(), invocation_id="i")
        self.assertEqual(transport.requests[0].target, "/resources/42")

    def test_path_traversal_resource_key_denied(self):
        gateway, _, _, broker, transport = make()
        with self.assertRaises(GatewayDenied):
            gateway.execute(self.request(key="../admin"), invocation_id="i")
        self.assertEqual(broker.scopes, [])
        self.assertEqual(transport.requests, [])

    def test_encoded_or_backslash_path_separators_denied(self):
        for key in ("%2fadmin", "a\\\\b", "x?y"):
            gateway, _, _, broker, transport = make()
            with self.assertRaises(GatewayDenied):
                gateway.execute(self.request(key=key), invocation_id="i")
            self.assertEqual(broker.scopes, [])
            self.assertEqual(transport.requests, [])

    def test_unknown_operation_denied(self):
        gateway, _, _, _, transport = make()
        with self.assertRaises(GatewayDenied):
            gateway.execute(self.request(operation="arbitrary-url"), invocation_id="i")
        self.assertEqual(transport.requests, [])

    def test_duplicate_invocation_never_reaches_transport_twice(self):
        gateway, _, _, _, transport = make()
        gateway.execute(self.request(), invocation_id="same")
        with self.assertRaises(GatewayDenied):
            gateway.execute(self.request(), invocation_id="same")
        self.assertEqual(len(transport.requests), 1)

    def test_credential_not_exposed_in_repr_or_receipt(self):
        gateway, _, _, _, transport = make()
        receipt = gateway.execute(self.request(), invocation_id="i")
        self.assertNotIn("scoped-secret", repr(transport.requests[0]))
        self.assertNotIn("scoped-secret", repr(receipt))

    def test_broker_overlong_ttl_denied(self):
        class BadBroker:
            def mint(self, scope):
                return ScopedCredential("too-long", 9999)
        gateway, _, _, _, transport = make()
        gateway.broker = BadBroker()
        with self.assertRaises(GatewayDenied):
            gateway.execute(self.request(), invocation_id="i")
        self.assertEqual(transport.requests, [])


if __name__ == "__main__":
    unittest.main()
