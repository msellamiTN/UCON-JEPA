---------------- MODULE G8_3_11_CrashConsistency ----------------
EXTENDS Naturals, FiniteSets

CONSTANTS EventId, EvaluationId

Phases == {"IDLE", "TXN", "PREPARED", "COMMITTED",
           "ACKNOWLEDGED", "RECOVERED_ABORT", "RECOVERED_COMMIT",
           "REPLAY_REJECTED"}

VARIABLES phase, durableEvents, durableEvaluations,
          stagedEvents, stagedEvaluations, acknowledged, commitCount

vars == <<phase, durableEvents, durableEvaluations,
          stagedEvents, stagedEvaluations, acknowledged, commitCount>>

Init ==
  /\ phase = "IDLE"
  /\ durableEvents = {}
  /\ durableEvaluations = {}
  /\ stagedEvents = {}
  /\ stagedEvaluations = {}
  /\ acknowledged = FALSE
  /\ commitCount = 0

Begin ==
  /\ phase = "IDLE"
  /\ phase' = "TXN"
  /\ UNCHANGED <<durableEvents, durableEvaluations,
       stagedEvents, stagedEvaluations, acknowledged, commitCount>>

Prepare ==
  /\ phase = "TXN"
  /\ phase' = "PREPARED"
  /\ stagedEvents' = {EventId}
  /\ stagedEvaluations' = {EvaluationId}
  /\ UNCHANGED <<durableEvents, durableEvaluations, acknowledged, commitCount>>

Commit ==
  /\ phase = "PREPARED"
  /\ durableEvents' = durableEvents \cup stagedEvents
  /\ durableEvaluations' = durableEvaluations \cup stagedEvaluations
  /\ stagedEvents' = {}
  /\ stagedEvaluations' = {}
  /\ commitCount' = commitCount + 1
  /\ phase' = "COMMITTED"
  /\ UNCHANGED acknowledged

Acknowledge ==
  /\ phase = "COMMITTED"
  /\ EventId \in durableEvents
  /\ EvaluationId \in durableEvaluations
  /\ acknowledged' = TRUE
  /\ phase' = "ACKNOWLEDGED"
  /\ UNCHANGED <<durableEvents, durableEvaluations,
       stagedEvents, stagedEvaluations, commitCount>>

CrashBeforeCommit ==
  /\ phase \in {"TXN", "PREPARED"}
  /\ phase' = "RECOVERED_ABORT"
  /\ stagedEvents' = {}
  /\ stagedEvaluations' = {}
  /\ UNCHANGED <<durableEvents, durableEvaluations, acknowledged, commitCount>>

CrashAfterCommitBeforeAck ==
  /\ phase = "COMMITTED"
  /\ phase' = "RECOVERED_COMMIT"
  /\ acknowledged' = FALSE
  /\ UNCHANGED <<durableEvents, durableEvaluations,
       stagedEvents, stagedEvaluations, commitCount>>

RetryAfterAbort ==
  /\ phase = "RECOVERED_ABORT"
  /\ phase' = "IDLE"
  /\ UNCHANGED <<durableEvents, durableEvaluations,
       stagedEvents, stagedEvaluations, acknowledged, commitCount>>

RetryCommittedEvent ==
  /\ phase = "RECOVERED_COMMIT"
  /\ EventId \in durableEvents
  /\ phase' = "REPLAY_REJECTED"
  /\ UNCHANGED <<durableEvents, durableEvaluations,
       stagedEvents, stagedEvaluations, acknowledged, commitCount>>

Next ==
  \/ Begin
  \/ Prepare
  \/ Commit
  \/ Acknowledge
  \/ CrashBeforeCommit
  \/ CrashAfterCommitBeforeAck
  \/ RetryAfterAbort
  \/ RetryCommittedEvent

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ phase \in Phases
  /\ durableEvents \subseteq {EventId}
  /\ durableEvaluations \subseteq {EvaluationId}
  /\ stagedEvents \subseteq {EventId}
  /\ stagedEvaluations \subseteq {EvaluationId}
  /\ acknowledged \in BOOLEAN
  /\ commitCount \in Nat

AtomicEventEvaluation ==
  (EventId \in durableEvents) <=> (EvaluationId \in durableEvaluations)

AtMostOnceCommit == commitCount <= 1

AcknowledgementRequiresCommit ==
  acknowledged =>
    /\ EventId \in durableEvents
    /\ EvaluationId \in durableEvaluations
    /\ commitCount = 1

ReplayRejectionRequiresPriorCommit ==
  phase = "REPLAY_REJECTED" =>
    /\ EventId \in durableEvents
    /\ EvaluationId \in durableEvaluations
    /\ commitCount = 1

\* Negative control: a faulty commit that persists only the event ID.
UnsafeCommitEventOnly ==
  /\ phase = "PREPARED"
  /\ durableEvents' = durableEvents \cup stagedEvents
  /\ durableEvaluations' = durableEvaluations
  /\ stagedEvents' = {}
  /\ stagedEvaluations' = {}
  /\ commitCount' = commitCount + 1
  /\ phase' = "COMMITTED"
  /\ UNCHANGED acknowledged

NextUnsafe == Next \/ UnsafeCommitEventOnly
SpecUnsafe == Init /\ [][NextUnsafe]_vars

=============================================================================
