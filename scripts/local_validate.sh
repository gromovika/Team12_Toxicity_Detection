#!/usr/bin/env bash
# Run from any directory: bash scripts/local_validate.sh <command>
# macOS/Linux, Docker Desktop/Engine with Compose v2, Python 3.12.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
TRACKING_URI="http://127.0.0.1:5000"
TEST_DB_URL='postgresql+psycopg2://toxicity_test:local_test_only_2026@127.0.0.1:55432/toxicity_test'
# API may use a different host port on macOS; Compose still talks to api:8000 internally.
# Never source .env: it also contains passwords and is not an executable script.
API_HOST_PORT="${API_HOST_PORT:-}"
if [[ -z "$API_HOST_PORT" && -f .env ]]; then
  API_HOST_PORT="$(sed -n 's/^API_HOST_PORT=//p' .env | tail -n 1 | tr -d '\r')"
fi
API_HOST_PORT="${API_HOST_PORT:-8000}"
if ! [[ "$API_HOST_PORT" =~ ^[0-9]+$ ]] || (( 10#$API_HOST_PORT < 1 || 10#$API_HOST_PORT > 65535 )); then
  echo 'Invalid API_HOST_PORT; choose an integer between 1 and 65535' >&2
  exit 1
fi
API_BASE_URL="http://127.0.0.1:${API_HOST_PORT}"

usage() {
  cat <<'TXT'
Available commands:
  setup        Create Python 3.12 virtual environment and install dependencies
  infra        Build and run API, PostgreSQL, test PostgreSQL, MLflow, Prometheus, Grafana
  status       Show container and service status
  tests        Run project tests + original FastAPI suite on dedicated real PostgreSQL
  mlflow       Import archived baseline into running MLflow and verify it
  train-fast   Train Naive Bayes on original dataset and log a candidate to MLflow
  train-all    Train all three algorithms (NB, LR, calibrated SVM), log champion
  compare      Print baseline vs last local candidate metrics
  clean-test   Remove ONLY disposable test PostgreSQL container

Run in this order: setup -> infra -> tests -> mlflow -> train-fast -> compare.
TXT
}

ensure_env() {
  if [[ -f .env ]]; then return; fi
  command -v python3 >/dev/null || { echo 'Install Python first'; exit 1; }
  python3 - <<'PY'
from pathlib import Path
import secrets
path = Path('.env')
example = Path('.env.example').read_text(encoding='utf-8')
example = example.replace('CHANGE_ME_TO_LONG_RANDOM_PASSWORD', secrets.token_hex(24), 1)
example = example.replace('CHANGE_ME_TO_LONG_RANDOM_PASSWORD', secrets.token_hex(24), 1)
path.write_text(example, encoding='utf-8')
path.chmod(0o600)
print('.env создан со случайными паролями (не добавляйте в git).')
PY
}

need_python() {
  if [[ ! -x "$PY" ]]; then
    echo 'First run: bash scripts/local_validate.sh setup'
    exit 1
  fi
}

need_docker() {
  command -v docker >/dev/null || { echo 'Docker not installed; install and start Docker Desktop/Engine.'; exit 1; }
  docker info >/dev/null || { echo 'Docker engine is not running. Start Docker Desktop.'; exit 1; }
  docker compose version >/dev/null || { echo 'Docker Compose v2 is missing.'; exit 1; }
}

wait_get() {
  local url="$1"
  local i
  for ((i=0;i<75;i++)); do
    if curl -fsS -o /dev/null "$url"; then return 0; fi
    sleep 2
  done
  echo "Service never became healthy: $url" >&2
  docker compose --profile test --profile mlflow ps
  exit 1
}

case "${1:-}" in
  setup)
    ensure_env
    command -v python3.12 >/dev/null || { echo 'Python 3.12 required; install it (for example, brew install python@3.12).'; exit 1; }
    python3.12 -m venv .venv
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -r requirements-local-check.txt
    "$PY" -m pip check
    ;;
  infra)
    ensure_env
    need_docker
    docker compose --profile mlflow --profile test config --quiet
    docker compose --profile mlflow --profile test up -d --build
    wait_get "$API_BASE_URL/health/ready"
    wait_get http://127.0.0.1:5000/
    wait_get http://127.0.0.1:9090/-/ready
    wait_get http://127.0.0.1:3000/login
    docker compose --profile test exec -T db_test pg_isready -U toxicity_test -d toxicity_test
    echo 'Docker API + main PostgreSQL + test PostgreSQL + MLflow + Prometheus + Grafana: running'
    ;;
  status)
    need_docker
    docker compose --profile test --profile mlflow ps
    curl -fsS "$API_BASE_URL/health/ready"; echo
    curl -fsS -X POST "$API_BASE_URL/predict" \
      -H 'Content-Type: application/json' --data '{"text":"Привет, как дела?"}'; echo
    curl -fsS "$API_BASE_URL/metrics" | grep '^toxicity_' | sed -n '1,15p'
    ;;
  tests)
    need_python
    need_docker
    docker compose --profile test exec -T db_test pg_isready -U toxicity_test -d toxicity_test
    "$PY" -m pytest -q tests
    PYTHONPATH="$ROOT/app:$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      TEAM12_TEST_DB_URL="$TEST_DB_URL" \
      "$PY" -m pytest -q app/tests
    echo 'All tests passed; app/tests used actual isolated Docker PostgreSQL.'
    ;;
  mlflow)
    need_python
    wait_get "$TRACKING_URI/"
    MLFLOW_TRACKING_URI="$TRACKING_URI" "$PY" -m mlops.import_baseline --tracking-uri "$TRACKING_URI"
    MLFLOW_TRACKING_URI="$TRACKING_URI" "$PY" -m mlops.verify_mlflow --tracking-uri "$TRACKING_URI"
    ;;
  train-fast)
    need_python
    [[ -f data/dataset.txt.zip ]] || { echo 'Original data/dataset.txt.zip not found; use the full project archive'; exit 1; }
    wait_get "$TRACKING_URI/"
    MLFLOW_TRACKING_URI="$TRACKING_URI" "$PY" -m mlops.train \
      --data data/dataset.txt.zip --models 'Naive Bayes' --output artifacts/candidate_model.joblib \
      --tracking-uri "$TRACKING_URI"
    MLFLOW_TRACKING_URI="$TRACKING_URI" "$PY" -m mlops.verify_mlflow --require-candidate
    ;;
  train-all)
    need_python
    [[ -f data/dataset.txt.zip ]] || { echo 'Original dataset missing'; exit 1; }
    wait_get "$TRACKING_URI/"
    MLFLOW_TRACKING_URI="$TRACKING_URI" "$PY" -m mlops.train \
      --data data/dataset.txt.zip --output artifacts/candidate_model.joblib \
      --tracking-uri "$TRACKING_URI"
    MLFLOW_TRACKING_URI="$TRACKING_URI" "$PY" -m mlops.verify_mlflow --require-candidate
    ;;
  compare)
    need_python
    "$PY" - <<'PY'
import json
from pathlib import Path
base = json.loads(Path('artifacts/baseline_metadata.json').read_text())
path = Path('artifacts/candidate_metadata.json')
if not path.exists():
    raise SystemExit('Train a candidate first (train-fast or train-all).')
cand = json.loads(path.read_text())
for k in ('test_macro_f1', 'test_toxic_recall'):
    print(f'{k}: baseline={base[k]:.6f}; candidate={cand[k]:.6f}')
if all(cand[k] >= base[k] for k in ('test_macro_f1', 'test_toxic_recall')):
    print('Candidate passes two metric checks. Manual review is still necessary.')
else:
    print('Candidate fails the baseline gate: DO NOT deploy over baseline.')
PY
    ;;
  clean-test)
    need_docker
    docker compose --profile test stop db_test
    docker compose --profile test rm -s -f -v db_test
    echo 'Only test PostgreSQL container removed. Production DB and named volumes retained.'
    ;;
  *) usage; exit 2;;
esac
