---------------- MODULE G6_RuntimeEnforcement ----------------
EXTENDS Naturals, FiniteSets

CONSTANTS Invocations, UCONAllowed,
          EnforceUCON, EnforceRevocation,
          EnforcePolicyFreshness, EnforceVerification

ASSUME /\ Invocations # {}
       /\ UCONAllowed \subseteq Invocations
       /\ EnforceUCON \in BOOLEAN
       /\ EnforceRevocation \in BOOLEAN
       /\ EnforcePolicyFreshness \in BOOLEAN
       /\ EnforceVerification \in BOOLEAN

Statuses == {"NEW", "AUTHORIZED", "DENIED", "DISPATCHED",
             "ACCEPTED", "OBSERVED", "VERIFIED",
             "COMMITTED", "UNKNOWN"}

JepaOutputs == {"ALLOW_RECOMMENDATION", "RESTRICT", "ESCALATE", "ABSTAIN"}

VARIABLES status, certPolicy, gatePassed, accepted,
          acceptedWhileRevoked, acceptedWithFreshPolicy,
          observed, verified, committed, outcomeUnknown,
          outcome, capRevoked, policyVersion, acceptedPolicy,
          jepa

vars == <<status, certPolicy, gatePassed, accepted,
          acceptedWhileRevoked, acceptedWithFreshPolicy,
          observed, verified, committed, outcomeUnknown,
          outcome, capRevoked, policyVersion, acceptedPolicy,
          jepa>>

Init ==
  /\ status = [i \in Invocations |-> "NEW"]
  /\ certPolicy = [i \in Invocations |-> 0]
  /\ gatePassed = [i \in Invocations |-> FALSE]
  /\ accepted = [i \in Invocations |-> FALSE]
  /\ acceptedWhileRevoked = [i \in Invocations |-> FALSE]
  /\ acceptedWithFreshPolicy = [i \in Invocations |-> FALSE]
  /\ observed = [i \in Invocations |-> FALSE]
  /\ verified = [i \in Invocations |-> FALSE]
  /\ committed = [i \in Invocations |-> FALSE]
  /\ outcomeUnknown = [i \in Invocations |-> FALSE]
  /\ outcome = [i \in Invocations |-> "NONE"]
  /\ capRevoked = FALSE
  /\ policyVersion = 1
  /\ acceptedPolicy = [i \in Invocations |-> 0]
  /\ jepa = [i \in Invocations |-> "ALLOW_RECOMMENDATION"]

Authorize(i) ==
  /\ i \in Invocations
  /\ status[i] = "NEW"
  /\ (\/~EnforceUCON \/ i \in UCONAllowed)
  /\ status' = [status EXCEPT ![i] = "AUTHORIZED"]
  /\ certPolicy' = [certPolicy EXCEPT ![i] = policyVersion]
  /\ UNCHANGED <<gatePassed, accepted, acceptedWhileRevoked,
       acceptedWithFreshPolicy, observed, verified, committed,
       outcomeUnknown, outcome, capRevoked, policyVersion,
       acceptedPolicy, jepa>>

Deny(i) ==
  /\ i \in Invocations
  /\ status[i] = "NEW"
  /\ EnforceUCON
  /\ i \notin UCONAllowed
  /\ status' = [status EXCEPT ![i] = "DENIED"]
  /\ UNCHANGED <<certPolicy, gatePassed, accepted,
       acceptedWhileRevoked, acceptedWithFreshPolicy, observed,
       verified, committed, outcomeUnknown, outcome,
       capRevoked, policyVersion, acceptedPolicy, jepa>>

GateCheck(i) ==
  /\ i \in Invocations
  /\ status[i] = "AUTHORIZED"
  /\ (\/~EnforceUCON \/ i \in UCONAllowed)
  /\ (\/~EnforcePolicyFreshness \/ certPolicy[i] = policyVersion)
  /\ (\/~EnforceRevocation \/ ~capRevoked)
  /\ gatePassed' = [gatePassed EXCEPT ![i] = TRUE]
  /\ status' = [status EXCEPT ![i] = "DISPATCHED"]
  /\ UNCHANGED <<certPolicy, accepted, acceptedWhileRevoked,
       acceptedWithFreshPolicy, observed, verified, committed,
       outcomeUnknown, outcome, capRevoked, policyVersion,
       acceptedPolicy, jepa>>

Accept(i) ==
  /\ i \in Invocations
  /\ status[i] = "DISPATCHED"
  /\ gatePassed[i]
  /\ i \in UCONAllowed
  /\ certPolicy[i] = policyVersion
  /\ ~capRevoked
  /\ accepted' = [accepted EXCEPT ![i] = TRUE]
  /\ acceptedWhileRevoked' =
       [acceptedWhileRevoked EXCEPT ![i] = capRevoked]
  /\ acceptedWithFreshPolicy' =
       [acceptedWithFreshPolicy EXCEPT ![i] = (certPolicy[i] = policyVersion)]
  /\ acceptedPolicy' = [acceptedPolicy EXCEPT ![i] = policyVersion]
  /\ status' = [status EXCEPT ![i] = "ACCEPTED"]
  /\ outcome' = [outcome EXCEPT ![i] = "ACCEPTED"]
  /\ UNCHANGED <<certPolicy, gatePassed, observed, verified,
       committed, outcomeUnknown, capRevoked, policyVersion, jepa>>

Observe(i) ==
  /\ i \in Invocations
  /\ status[i] = "ACCEPTED"
  /\ accepted[i]
  /\ observed' = [observed EXCEPT ![i] = TRUE]
  /\ status' = [status EXCEPT ![i] = "OBSERVED"]
  /\ UNCHANGED <<certPolicy, gatePassed, accepted,
       acceptedWhileRevoked, acceptedWithFreshPolicy, verified,
       committed, outcomeUnknown, outcome, capRevoked,
       policyVersion, acceptedPolicy, jepa>>

Verify(i) ==
  /\ i \in Invocations
  /\ status[i] = "OBSERVED"
  /\ observed[i]
  /\ verified' = [verified EXCEPT ![i] = TRUE]
  /\ status' = [status EXCEPT ![i] = "VERIFIED"]
  /\ UNCHANGED <<certPolicy, gatePassed, accepted,
       acceptedWhileRevoked, acceptedWithFreshPolicy, observed,
       committed, outcomeUnknown, outcome, capRevoked,
       policyVersion, acceptedPolicy, jepa>>

Commit(i) ==
  /\ i \in Invocations
  /\ \/ /\ status[i] = "VERIFIED"
        /\ verified[i]
        /\ observed[i]
     \/ /\ ~EnforceVerification
        /\ status[i] = "OBSERVED"
        /\ observed[i]
  /\ committed' = [committed EXCEPT ![i] = TRUE]
  /\ status' = [status EXCEPT ![i] = "COMMITTED"]
  /\ outcome' = [outcome EXCEPT ![i] = "COMMITTED"]
  /\ UNCHANGED <<certPolicy, gatePassed, accepted,
       acceptedWhileRevoked, acceptedWithFreshPolicy, observed,
       verified, outcomeUnknown, capRevoked, policyVersion,
       acceptedPolicy, jepa>>

TimeoutUnknown(i) ==
  /\ i \in Invocations
  /\ status[i] = "DISPATCHED"
  /\ status' = [status EXCEPT ![i] = "UNKNOWN"]
  /\ outcomeUnknown' = [outcomeUnknown EXCEPT ![i] = TRUE]
  /\ outcome' = [outcome EXCEPT ![i] = "UNKNOWN"]
  /\ UNCHANGED <<certPolicy, gatePassed, accepted,
       acceptedWhileRevoked, acceptedWithFreshPolicy, observed,
       verified, committed, capRevoked, policyVersion,
       acceptedPolicy, jepa>>


LoseAcceptanceResponse(i) ==
  /\ i \in Invocations
  /\ status[i] = "ACCEPTED"
  /\ accepted[i]
  /\ status' = [status EXCEPT ![i] = "UNKNOWN"]
  /\ outcomeUnknown' = [outcomeUnknown EXCEPT ![i] = TRUE]
  /\ outcome' = [outcome EXCEPT ![i] = "UNKNOWN"]
  /\ UNCHANGED <<certPolicy, gatePassed, accepted,
       acceptedWhileRevoked, acceptedWithFreshPolicy, observed,
       verified, committed, capRevoked, policyVersion,
       acceptedPolicy, jepa>>

EffectiveRevoke ==
  /\ ~capRevoked
  /\ capRevoked' = TRUE
  /\ UNCHANGED <<status, certPolicy, gatePassed, accepted,
       acceptedWhileRevoked, acceptedWithFreshPolicy, observed,
       verified, committed, outcomeUnknown, outcome,
       policyVersion, acceptedPolicy, jepa>>

PublishPolicy ==
  /\ policyVersion = 1
  /\ policyVersion' = 2
  /\ UNCHANGED <<status, certPolicy, gatePassed, accepted,
       acceptedWhileRevoked, acceptedWithFreshPolicy, observed,
       verified, committed, outcomeUnknown, outcome,
       capRevoked, acceptedPolicy, jepa>>

ChangeJepa(i) ==
  /\ i \in Invocations
  /\ \/ /\ jepa[i] # "ALLOW_RECOMMENDATION"
        /\ jepa' = [jepa EXCEPT ![i] = "ALLOW_RECOMMENDATION"]
     \/ /\ jepa[i] # "RESTRICT"
        /\ jepa' = [jepa EXCEPT ![i] = "RESTRICT"]
     \/ /\ jepa[i] # "ESCALATE"
        /\ jepa' = [jepa EXCEPT ![i] = "ESCALATE"]
     \/ /\ jepa[i] # "ABSTAIN"
        /\ jepa' = [jepa EXCEPT ![i] = "ABSTAIN"]
  /\ UNCHANGED <<status, certPolicy, gatePassed, accepted,
       acceptedWhileRevoked, acceptedWithFreshPolicy, observed,
       verified, committed, outcomeUnknown, outcome,
       capRevoked, policyVersion, acceptedPolicy>>

Next ==
  \/ \E i \in Invocations : Authorize(i)
  \/ \E i \in Invocations : Deny(i)
  \/ \E i \in Invocations : GateCheck(i)
  \/ \E i \in Invocations : Accept(i)
  \/ \E i \in Invocations : Observe(i)
  \/ \E i \in Invocations : Verify(i)
  \/ \E i \in Invocations : Commit(i)
  \/ \E i \in Invocations : TimeoutUnknown(i)
  \/ \E i \in Invocations : LoseAcceptanceResponse(i)
  \/ EffectiveRevoke
  \/ PublishPolicy
  \/ \E i \in Invocations : ChangeJepa(i)

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ status \in [Invocations -> Statuses]
  /\ certPolicy \in [Invocations -> {0, 1, 2}]
  /\ gatePassed \in [Invocations -> BOOLEAN]
  /\ accepted \in [Invocations -> BOOLEAN]
  /\ acceptedWhileRevoked \in [Invocations -> BOOLEAN]
  /\ acceptedWithFreshPolicy \in [Invocations -> BOOLEAN]
  /\ observed \in [Invocations -> BOOLEAN]
  /\ verified \in [Invocations -> BOOLEAN]
  /\ committed \in [Invocations -> BOOLEAN]
  /\ outcomeUnknown \in [Invocations -> BOOLEAN]
  /\ outcome \in [Invocations -> {"NONE", "ACCEPTED", "UNKNOWN", "COMMITTED"}]
  /\ capRevoked \in BOOLEAN
  /\ policyVersion \in {1, 2}
  /\ acceptedPolicy \in [Invocations -> {0, 1, 2}]
  /\ jepa \in [Invocations -> JepaOutputs]

NoUnauthorizedAcceptance ==
  \A i \in Invocations : accepted[i] => i \in UCONAllowed

NoAcceptanceAfterRevocation ==
  \A i \in Invocations : accepted[i] => ~acceptedWhileRevoked[i]

NoStalePolicyAcceptance ==
  \A i \in Invocations : accepted[i] => acceptedWithFreshPolicy[i]

CommitRequiresVerification ==
  \A i \in Invocations : committed[i] => verified[i] /\ observed[i]

UnknownNotCommitted ==
  \A i \in Invocations : outcomeUnknown[i] => ~committed[i]

JepaCannotAuthorize ==
  \A i \in Invocations : accepted[i] => i \in UCONAllowed

=============================================================================
