---------------- MODULE G8_3_12_MultiWriterReplay ----------------
EXTENDS Naturals, FiniteSets

CONSTANTS Writers, DuplicateMode

ASSUME /\ Writers = {"w1", "w2"}
       /\ DuplicateMode \in BOOLEAN

EventIdByWriter ==
  [w \in Writers |->
    IF DuplicateMode THEN "event-1"
    ELSE IF w = "w1" THEN "event-1" ELSE "event-2"]

EvaluationIdByWriter ==
  [w \in Writers |->
    IF w = "w1" THEN "evaluation-1" ELSE "evaluation-2"]

EventIds == {EventIdByWriter[w] : w \in Writers}
EvaluationIds == {EvaluationIdByWriter[w] : w \in Writers}
Phases == {"READY", "LOCKED", "PREPARED", "COMMITTED",
           "ACKNOWLEDGED", "ABORTED", "RECOVERED_COMMIT",
           "REPLAY_REJECTED"}

VARIABLES phase, lockOwner, durablePairs, stagedPair

vars == <<phase, lockOwner, durablePairs, stagedPair>>

Init ==
  /\ phase = [w \in Writers |-> "READY"]
  /\ lockOwner = "NONE"
  /\ durablePairs = {}
  /\ stagedPair = "NONE"

Acquire(w) ==
  /\ w \in Writers
  /\ phase[w] = "READY"
  /\ lockOwner = "NONE"
  /\ lockOwner' = w
  /\ phase' = [phase EXCEPT ![w] = "LOCKED"]
  /\ UNCHANGED <<durablePairs, stagedPair>>

Prepare(w) ==
  /\ w \in Writers
  /\ lockOwner = w
  /\ phase[w] = "LOCKED"
  /\ EventIdByWriter[w] \notin {p[1] : p \in durablePairs}
  /\ stagedPair' = <<EventIdByWriter[w], EvaluationIdByWriter[w]>>
  /\ phase' = [phase EXCEPT ![w] = "PREPARED"]
  /\ UNCHANGED <<lockOwner, durablePairs>>

RejectReplay(w) ==
  /\ w \in Writers
  /\ lockOwner = w
  /\ phase[w] = "LOCKED"
  /\ EventIdByWriter[w] \in {p[1] : p \in durablePairs}
  /\ phase' = [phase EXCEPT ![w] = "REPLAY_REJECTED"]
  /\ lockOwner' = "NONE"
  /\ UNCHANGED <<durablePairs, stagedPair>>

Commit(w) ==
  /\ w \in Writers
  /\ lockOwner = w
  /\ phase[w] = "PREPARED"
  /\ stagedPair = <<EventIdByWriter[w], EvaluationIdByWriter[w]>>
  /\ stagedPair \notin durablePairs
  /\ durablePairs' = durablePairs \cup {stagedPair}
  /\ phase' = [phase EXCEPT ![w] = "COMMITTED"]
  /\ lockOwner' = "NONE"
  /\ stagedPair' = "NONE"

Acknowledge(w) ==
  /\ w \in Writers
  /\ phase[w] = "COMMITTED"
  /\ <<EventIdByWriter[w], EvaluationIdByWriter[w]>> \in durablePairs
  /\ phase' = [phase EXCEPT ![w] = "ACKNOWLEDGED"]
  /\ UNCHANGED <<lockOwner, durablePairs, stagedPair>>

CrashBeforeCommit(w) ==
  /\ w \in Writers
  /\ lockOwner = w
  /\ phase[w] \in {"LOCKED", "PREPARED"}
  /\ phase' = [phase EXCEPT ![w] = "ABORTED"]
  /\ lockOwner' = "NONE"
  /\ stagedPair' = "NONE"
  /\ UNCHANGED durablePairs

RetryAfterAbort(w) ==
  /\ w \in Writers
  /\ phase[w] = "ABORTED"
  /\ phase' = [phase EXCEPT ![w] = "READY"]
  /\ UNCHANGED <<lockOwner, durablePairs, stagedPair>>

CrashAfterCommitBeforeAck(w) ==
  /\ w \in Writers
  /\ phase[w] = "COMMITTED"
  /\ phase' = [phase EXCEPT ![w] = "RECOVERED_COMMIT"]
  /\ UNCHANGED <<lockOwner, durablePairs, stagedPair>>

RetryCommittedEvent(w) ==
  /\ w \in Writers
  /\ phase[w] = "RECOVERED_COMMIT"
  /\ <<EventIdByWriter[w], EvaluationIdByWriter[w]>> \in durablePairs
  /\ phase' = [phase EXCEPT ![w] = "REPLAY_REJECTED"]
  /\ UNCHANGED <<lockOwner, durablePairs, stagedPair>>

Terminal ==
  /\ \E w \in Writers : phase[w] \in {"ACKNOWLEDGED", "REPLAY_REJECTED"}
  /\ UNCHANGED vars

Next ==
  \/ \E w \in Writers : Acquire(w)
  \/ \E w \in Writers : Prepare(w)
  \/ \E w \in Writers : RejectReplay(w)
  \/ \E w \in Writers : Commit(w)
  \/ \E w \in Writers : Acknowledge(w)
  \/ \E w \in Writers : CrashBeforeCommit(w)
  \/ \E w \in Writers : RetryAfterAbort(w)
  \/ \E w \in Writers : CrashAfterCommitBeforeAck(w)
  \/ \E w \in Writers : RetryCommittedEvent(w)
  \/ Terminal

Spec == Init /\ [][Next]_vars

DurableEvents == {p[1] : p \in durablePairs}
DurableEvaluations == {p[2] : p \in durablePairs}

TypeOK ==
  /\ phase \in [Writers -> Phases]
  /\ lockOwner \in Writers \cup {"NONE"}
  /\ durablePairs \subseteq (EventIds \X EvaluationIds)
  /\ stagedPair \in {"NONE"} \cup (EventIds \X EvaluationIds)

NoDuplicateEventCommit ==
  \A p1, p2 \in durablePairs : p1[1] = p2[1] => p1 = p2

CommitRecordWellFormed ==
  \A p \in durablePairs :
    \E w \in Writers :
      p = <<EventIdByWriter[w], EvaluationIdByWriter[w]>>

AcknowledgementRequiresDurability ==
  \A w \in Writers :
    phase[w] = "ACKNOWLEDGED" =>
      <<EventIdByWriter[w], EvaluationIdByWriter[w]>> \in durablePairs

ReplayRejectionRequiresPriorCommit ==
  \A w \in Writers :
    phase[w] = "REPLAY_REJECTED" =>
      EventIdByWriter[w] \in DurableEvents

LockConsistency ==
  /\ (lockOwner = "NONE" => stagedPair = "NONE")
  /\ (lockOwner \in Writers =>
       phase[lockOwner] \in {"LOCKED", "PREPARED"})
  /\ (lockOwner \in Writers /\ phase[lockOwner] = "PREPARED" =>
       stagedPair = <<EventIdByWriter[lockOwner], EvaluationIdByWriter[lockOwner]>>)
  /\ (lockOwner \in Writers /\ phase[lockOwner] = "LOCKED" =>
       stagedPair = "NONE")

\* Negative control: bypass replay rejection and commit a second
\* evaluation for an event ID that is already durable.
UnsafeDuplicateCommit(w) ==
  /\ w \in Writers
  /\ lockOwner = w
  /\ phase[w] = "LOCKED"
  /\ EventIdByWriter[w] \in DurableEvents
  /\ durablePairs' = durablePairs \cup
       {<<EventIdByWriter[w], EvaluationIdByWriter[w]>>}
  /\ phase' = [phase EXCEPT ![w] = "COMMITTED"]
  /\ lockOwner' = "NONE"
  /\ stagedPair' = "NONE"

NextUnsafe == Next \/ (\E w \in Writers : UnsafeDuplicateCommit(w))
SpecUnsafe == Init /\ [][NextUnsafe]_vars

=============================================================================
