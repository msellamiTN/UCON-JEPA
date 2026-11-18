# G7.8 PostgreSQL HA lab (Patroni + etcd)

**Status: Compose model and Patroni image build passed CI. The first live smoke-test attempt exposed missing replication HBA rules and a mounted data-directory layout issue; both were corrected, and the rerun is pending. No runtime HA acceptance is claimed yet.**

This lab is intentionally separate from the single-primary PostgreSQL CI service. It provisions three PostgreSQL/Patroni nodes, a three-member etcd quorum, and HAProxy routing to the Patroni primary.

## Components

- PostgreSQL 16.6
- Patroni 4.0.5
- etcd 3.5.17 (three members)
- HAProxy 3.0.7
- Independent named volumes for each PostgreSQL and etcd member

The image tags are pinned. This is a local experimental topology, not a production-ready deployment: TLS, secrets management, resource limits, host-level failure domains, and secure network segmentation remain deployment hardening tasks.

## Start

From this directory:

```bash
cp .env.example .env
# Edit .env and replace both passwords.
docker compose -f compose.yaml config
docker compose -f compose.yaml build
docker compose -f compose.yaml up -d
docker compose -f compose.yaml ps
```

Check DCS health:

```bash
docker compose -f compose.yaml exec etcd1 \
  etcdctl --endpoints=http://etcd1:2379,http://etcd2:2379,http://etcd3:2379 endpoint health
```

Inspect cluster state:

```bash
docker compose -f compose.yaml exec pg1 \
  patronictl -c /etc/patroni.yml list
```

Connect through HAProxy on localhost port 5000 using the `ucon_admin` role and the password in `.env`. HAProxy routes to the node whose Patroni REST endpoint reports `/primary`.

## Configure synchronous replication

The default bootstrap configuration is asynchronous. Record this baseline first. Then enable synchronous mode using Patronictl:

```bash
docker compose -f compose.yaml exec pg1 \
  patronictl -c /etc/patroni.yml edit-config ucon-jepa-ha
```

Set the DCS configuration fields to:

```yaml
synchronous_mode: true
synchronous_mode_strict: true
synchronous_node_count: 1
postgresql:
  parameters:
    synchronous_commit: "on"
```

Confirm the effective DCS configuration and PostgreSQL runtime state before running a trial. Record `pg_stat_replication`, `pg_stat_wal_receiver`, `pg_current_wal_lsn()`, and `pg_last_wal_replay_lsn()` from the relevant nodes. Reset the topology between async and sync campaigns so that the two modes do not share state.

With strict synchronous mode, writes may block if no synchronous standby is available. This is expected durability/availability behavior and must be measured.

## Controlled failure exercises

Do not run destructive campaigns until a baseline is captured and the cluster is healthy.

Primary process crash:

```bash
docker compose -f compose.yaml stop pg1
docker compose -f compose.yaml exec pg2 patronictl -c /etc/patroni.yml list
```

Restore the former primary only after recording the promoted leader and its timeline:

```bash
docker compose -f compose.yaml start pg1
docker compose -f compose.yaml exec pg2 patronictl -c /etc/patroni.yml list
```

A container stop is a process/node failure experiment; it is **not** a network partition experiment.

## Network partition campaign

The Compose bridge network is named `ucon-jepa-ha_ha` by default. Use a dedicated fault-injection harness to isolate a node from the DCS and client path while preserving its PostgreSQL process and storage. Record the exact network rules and timestamps. Never claim split-brain exclusion from a simple `docker stop` test.

For every partition trial, test the former primary directly as well as through HAProxy. If it can commit a write to divergent storage after the controller considers it fenced, record a safety counterexample. Do not promote a standby unless the controller's fencing and promotion state is recorded.

## Evidence capture

Each trial must preserve:
- a unique trial ID and random seed;
- compose config, image digests, Patroni and etcd configuration;
- DCS membership and Patroni cluster snapshots before/during/after;
- node identity, PostgreSQL timeline, WAL LSNs, synchronous standby state;
- client invocation IDs, request/response timestamps and observed outcomes;
- controller, HAProxy, PostgreSQL, and fault-injector logs;
- RPO/RTO calculations and unresolved/UNKNOWN outcomes.

Use the G7.8 protocol at `docs/research/G7.8_MULTI_NODE_POSTGRESQL_HA_EXPERIMENT_PROTOCOL.md` for campaigns and acceptance criteria.

## Teardown

After exporting all evidence:

```bash
docker compose -f compose.yaml down
```

To reset all database state for a new replication-mode campaign, and only after evidence has been archived:

```bash
docker compose -f compose.yaml down -v
```

## Limitations

- This Compose topology places all nodes on one Docker host and one bridge network; it does not model independent availability zones or host failures.
- The first CI validation confirmed Compose syntax and image build. The first runtime attempt started the containers but failed to establish streaming replication: PostgreSQL reported no matching `pg_hba.conf` entry for the replication user. Patroni also could not rename the data directory because it was mounted at the mount-point root. The configuration now adds explicit replication/client HBA rules and moves `data_dir` into a child directory of the mounted volume. The corrected runtime rerun is pending.\n- The lab has not been partition-tested or subjected to a controlled promotion campaign.
- Patroni/etcd election is not by itself proof of safe storage fencing under arbitrary partitions.
- Do not treat the writer_epoch field as an external fencing mechanism. The former primary must be demonstrably unable to accept protected writes.
- No G7.8 RPO, RTO, split-brain, or HA result is claimed until raw trial evidence is committed.
