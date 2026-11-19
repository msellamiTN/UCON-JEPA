---------------- MODULE G7_WriterFencing ----------------
EXTENDS Naturals, FiniteSets, TLC

CONSTANT AllowUnsafePromotion

VARIABLES phase, oldWritable, newWritable, oldEpoch, newEpoch

vars == <<phase, oldWritable, newWritable, oldEpoch, newEpoch>>

Init ==
    /\ phase = "old"
    /\ oldWritable = TRUE
    /\ newWritable = FALSE
    /\ oldEpoch = 1
    /\ newEpoch = 1

FenceOldPrimary ==
    /\ phase = "old"
    /\ oldWritable
    /\ oldWritable' = FALSE
    /\ UNCHANGED <<phase, newWritable, oldEpoch, newEpoch>>

Promote ==
    /\ phase = "old"
    /\ (AllowUnsafePromotion \/ ~oldWritable)
    /\ phase' = "new"
    /\ newWritable' = TRUE
    /\ newEpoch' = oldEpoch + 1
    /\ UNCHANGED <<oldWritable, oldEpoch>>

Next ==
    FenceOldPrimary \/ Promote

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ phase \in {"old", "new"}
    /\ oldWritable \in BOOLEAN
    /\ newWritable \in BOOLEAN
    /\ oldEpoch \in Nat
    /\ newEpoch \in Nat

WritablePrimaries ==
    {node \in {"old", "new"} :
        IF node = "old" THEN oldWritable ELSE newWritable}

AtMostOneWritablePrimary ==
    Cardinality(WritablePrimaries) <= 1

PromotionAdvancesEpoch ==
    newWritable => newEpoch > oldEpoch

OldPrimaryFencedBeforePromotion ==
    newWritable => ~oldWritable

====
