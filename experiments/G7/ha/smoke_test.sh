#!/usr/bin/env bash
set -Eeuo pipefail

dc=(docker compose -f compose.yaml)
: "${PATRONI_SUPERUSER_PASSWORD:=local-only-change-me}"

cleanup() {
  "${dc[@]}" logs --no-color > ha-smoke-compose.log 2>&1 || true
  if [ "${KEEP_CLUSTER:-0}" != "1" ]; then
    "${dc[@]}" down -v --remove-orphans || true
  else
    echo "KEEP_CLUSTER=1: preserving healthy cluster for subsequent experiment"
  fi
}
trap cleanup EXIT

echo "[1/5] Waiting for the three-member etcd quorum"
healthy=0
for attempt in $(seq 1 40); do
  if "${dc[@]}" exec -T etcd1 etcdctl \
      --endpoints=http://etcd1:2379,http://etcd2:2379,http://etcd3:2379 endpoint health >/dev/null 2>&1; then
    healthy=1
    break
  fi
  sleep 3
done
test "$healthy" -eq 1

echo "[2/5] Waiting for exactly one Patroni primary"
leader_count=0
for attempt in $(seq 1 60); do
  leader_count=0
  for node in pg1 pg2 pg3; do
    if "${dc[@]}" exec -T "$node" python3 -c \
      'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8008/primary", timeout=2)' \
      >/dev/null 2>&1; then
      leader_count=$((leader_count + 1))
    fi
  done
  if [ "$leader_count" -eq 1 ]; then break; fi
  sleep 3
done
if [ "$leader_count" -ne 1 ]; then
  echo "Expected exactly one /primary endpoint; observed $leader_count" >&2
  "${dc[@]}" exec -T pg1 patronictl -c /etc/patroni.yml list || true
  exit 1
fi

echo "[3/5] Waiting for both standbys to stream"
replicas=0
for attempt in $(seq 1 40); do
  replicas=$("${dc[@]}" exec -T pg1 env PGPASSWORD="$PATRONI_SUPERUSER_PASSWORD" \
    psql -h haproxy -p 5000 -U ucon_admin -d postgres -Atqc \
    "SELECT count(*) FROM pg_stat_replication" 2>/dev/null | tr -d '\r' || echo 0)
  if [ "$replicas" -ge 2 ] 2>/dev/null; then break; fi
  sleep 3
done
if [ "$replicas" -lt 2 ]; then
  echo "Expected two streaming standbys; observed $replicas" >&2
  exit 1
fi

echo "[4/5] Verifying HAProxy write routing and database read-back"
trial_id="github-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}"
"${dc[@]}" exec -T pg1 env PGPASSWORD="$PATRONI_SUPERUSER_PASSWORD" \
  psql -h haproxy -p 5000 -U ucon_admin -d postgres -v ON_ERROR_STOP=1 \
  -c "CREATE TABLE IF NOT EXISTS g78_smoke (trial_id text PRIMARY KEY, committed_at timestamptz NOT NULL DEFAULT now())" \
  -c "INSERT INTO g78_smoke(trial_id) VALUES ('$trial_id') ON CONFLICT (trial_id) DO NOTHING" \
  -c "SELECT trial_id FROM g78_smoke WHERE trial_id = '$trial_id'"

echo "[5/5] Capturing cluster topology"
"${dc[@]}" exec -T pg1 patronictl -c /etc/patroni.yml list
echo "G7.8 cluster smoke test passed. This is startup/replication/routing evidence only; no failover or partition guarantee is inferred."
