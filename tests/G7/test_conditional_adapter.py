import unittest

from ucon_jepa.conditional_adapter import (
    AdapterResult, ConditionalAdapter, GateDecision, HttpResponse,
    Mutation, Outcome, TransportFailure,
)


class FakeTransport:
    def __init__(self, response=None, failure=None):
        self.response = response
        self.failure = failure
        self.requests = []

    def send(self, request):
        self.requests.append(request)
        if self.failure:
            raise self.failure
        return self.response


def decision(**overrides):
    fields = dict(
        invocation_id="inv-001", effective_principal="svc-agent",
        method="PATCH", target="/v1/resources/42", capability="resource:update",
        policy_version=7, authorization_epoch=11, resource_etag='"v3"', allowed=True,
    )
    fields.update(overrides)
    return GateDecision(**fields)


class ConditionalAdapterTests(unittest.TestCase):
    def test_success_is_acceptance_not_verification(self):
        transport = FakeTransport(HttpResponse(204, {"X-Receipt-ID": "r-1"}))
        result = ConditionalAdapter(transport).mutate(decision(), Mutation(b'{"x":1}'))
        self.assertEqual(result.outcome, Outcome.ACCEPTED)
        self.assertEqual(transport.requests[0].headers["If-Match"], '"v3"')
        self.assertEqual(transport.requests[0].headers["X-UCON-Authorization-Epoch"], "11")
        self.assertIn("verification required", result.reason)

    def test_denial_never_calls_transport(self):
        transport = FakeTransport(HttpResponse(204, {"X-Receipt-ID": "r"}))
        result = ConditionalAdapter(transport).mutate(decision(allowed=False), Mutation(b"{}"))
        self.assertEqual(result.outcome, Outcome.REJECTED)
        self.assertEqual(transport.requests, [])

    def test_weak_etag_rejected(self):
        transport = FakeTransport(HttpResponse(204, {"X-Receipt-ID": "r"}))
        result = ConditionalAdapter(transport).mutate(decision(resource_etag='W/"v3"'), Mutation(b"{}"))
        self.assertEqual(result.outcome, Outcome.REJECTED)
        self.assertEqual(transport.requests, [])

    def test_missing_etag_rejected(self):
        transport = FakeTransport(HttpResponse(204, {"X-Receipt-ID": "r"}))
        result = ConditionalAdapter(transport).mutate(decision(resource_etag=""), Mutation(b"{}"))
        self.assertEqual(result.outcome, Outcome.REJECTED)
        self.assertEqual(transport.requests, [])

    def test_precondition_failure_is_conflict_without_retry(self):
        transport = FakeTransport(HttpResponse(412, {}))
        result = ConditionalAdapter(transport).mutate(decision(), Mutation(b"{}"))
        self.assertEqual(result.outcome, Outcome.CONFLICT)
        self.assertEqual(len(transport.requests), 1)

    def test_ambiguous_transport_failure_is_unknown_without_retry(self):
        transport = FakeTransport(failure=TransportFailure("connection lost"))
        result = ConditionalAdapter(transport).mutate(decision(), Mutation(b"{}"))
        self.assertEqual(result.outcome, Outcome.UNKNOWN)
        self.assertEqual(len(transport.requests), 1)

    def test_success_without_receipt_is_unknown(self):
        transport = FakeTransport(HttpResponse(200, {}))
        result = ConditionalAdapter(transport).mutate(decision(), Mutation(b"{}"))
        self.assertEqual(result.outcome, Outcome.UNKNOWN)

    def test_duplicate_invocation_is_rejected_locally(self):
        transport = FakeTransport(HttpResponse(204, {"X-Receipt-ID": "r"}))
        adapter = ConditionalAdapter(transport)
        adapter.mutate(decision(), Mutation(b"{}"))
        second = adapter.mutate(decision(), Mutation(b"{}"))
        self.assertEqual(second.outcome, Outcome.REJECTED)
        self.assertEqual(len(transport.requests), 1)

    def test_method_and_binding_validation(self):
        transport = FakeTransport(HttpResponse(204, {"X-Receipt-ID": "r"}))
        bad_method = ConditionalAdapter(transport).mutate(decision(method="GET"), Mutation(b"{}"))
        self.assertEqual(bad_method.outcome, Outcome.REJECTED)
        self.assertEqual(transport.requests, [])

    def test_remote_auth_rejection(self):
        transport = FakeTransport(HttpResponse(403, {}))
        result = ConditionalAdapter(transport).mutate(decision(), Mutation(b"{}"))
        self.assertEqual(result.outcome, Outcome.REJECTED)


if __name__ == "__main__":
    unittest.main()
