#!/usr/bin/env bash
# Disposable, single-host process-crash experiment. Not a partition/split-brain test.
set -Eeuo pipefail

dc=(docker compose -f compose.yaml)
: "${PATRONI_SUPERUSER_PASSWORD:=local-only-change-me}"
trial_id="crash-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}-$(date -u +%Y%m%dT%H%M%SZ)"
result_file="ha-primary-crash-results.env"
started_epoch="$(date +%s)"

primary_nodes() {
  local node
  for node in pg1 pg2 pg3; do
    if "${dc[@]}" exec -T "$node" python3 -c \
      'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8008/primary", timeout=2)' \
      >/dev/null 2>&1; then
      printf '%s\n' "$node"
    fi
  done
}

psql_haproxy() {
  local node
  for node in pg1 pg2 pg3; do
    if output=$("${dc[@]}" exec -T "$node" env PGPASSWORD="$PATRONI_SUPERUSER_PASSWORD" \
      psql -h haproxy -p 5000 -U ucon_admin -d postgres -v ON_ERROR_STOP=1 "$@" 2>/dev/null); then
      printf '%s\n' "$output"
      return 0
    fi
  done
  echo "No running PostgreSQL container could reach HAProxy" >&2
  return 1
}

echo "[1/6] Confirm exactly one primary before fault injection"
mapfile -t before_leaders < <(primary_nodes)
if [ "${#before_leaders[@]}" -ne 1 ]; then
  echo "Refusing crash experiment: expected one primary, observed ${#before_leaders[@]}" >&2
  "${dc[@]}" exec -T pg1 patronictl -c /etc/patroni.yml list || true
  exit 1
fi
old_primary="${before_leaders[0]}"
echo "old_primary=$old_primary"

echo "[2/6] Commit a unique invocation record through HAProxy"
psql_haproxy -c "CREATE TABLE IF NOT EXISTS g78_crash_probe (trial_id text PRIMARY KEY, committed_at timestamptz NOT NULL DEFAULT clock_timestamp())" \
  -c "INSERT INTO g78_crash_probe(trial_id) VALUES ('$trial_id')" \
  -c "SELECT trial_id, committed_at FROM g78_crash_probe WHERE trial_id='$trial_id'"

echo "[3/6] Stop the former primary process"
fault_start="$(date +%s)"
"${dc[@]}" stop "$old_primary"

echo "[4/6] Wait for a different, unique Patroni primary"
new_primary=""
for attempt in $(seq 1 60); do
  mapfile -t leaders < <(primary_nodes)
  if [ "${#leaders[@]}" -eq 1 ] && [ "${leaders[0]}" != "$old_primary" ]; then
    new_primary="${leaders[0]}"
    break
  fi
  sleep 2
done
if [ -z "$new_primary" ]; then
  echo "No unique replacement primary observed within 120 seconds" >&2
  "${dc[@]}" exec -T pg2 patronictl -c /etc/patroni.yml list || true
  exit 1
fi
promoted_epoch="$(date +%s)"
rto_seconds=$((promoted_epoch - fault_start))

echo "[5/6] Verify acknowledged record through the new HAProxy route"
observed="$(psql_haproxy -Atqc "SELECT count(*) FROM g78_crash_probe WHERE trial_id='$trial_id'")"
if [ "$observed" != "1" ]; then
  echo "Acknowledged record missing after promotion; observed count=$observed" >&2
  exit 1
fi

echo "[6/6] Capture promotion state and measured result"
{
  echo "trial_id=$trial_id"
  echo "old_primary=$old_primary"
  echo "new_primary=$new_primary"
  echo "fault_start_epoch=$fault_start"
  echo "promotion_observed_epoch=$promoted_epoch"
  echo "observed_rto_seconds=$rto_seconds"
  echo "acknowledged_record_visible=true"
  echo "experiment=process_crash_only"
  echo "started_epoch=$started_epoch"
} | tee "$result_file"
"${dc[@]}" exec -T "$new_primary" patronictl -c /etc/patroni.yml list
echo "Process-crash experiment passed. This does not test network partition, old-primary direct writes, split-brain exclusion, or async/sync durability bounds."
