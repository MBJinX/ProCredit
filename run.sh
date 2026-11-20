#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
usage() {
  echo "Usage: ./run.sh [setup|run|train|ui]   (settings: .env or ENV_FILE)"
}
case "${1:-run}" in
  -h|--help) usage; exit 0 ;;
  setup|run|train|ui) ;;
  *) usage >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then usage >&2; exit 2; fi
if [[ ! -f ${ENV_FILE:-.env} ]]; then
  echo "Copy .env.example to .env and edit it first." >&2; exit 2
fi
set -a
source "${ENV_FILE:-.env}"
set +a
: "${APPWORLD_PYTHON:=python}"
export PYTHONPATH="$PWD/verl/recipe${PYTHONPATH:+:$PYTHONPATH}"
case "${1:-run}" in
  ui) exec "$APPWORLD_PYTHON" -m http.server "${UI_PORT:-8080}" --bind 127.0.0.1 --directory "$PWD/ui" ;;
  setup)
    : "${APPWORLD_ROOT:?Set APPWORLD_ROOT to a directory on local disk}"
    if [[ -e "$APPWORLD_ROOT/data" ]]; then
      echo "Data directory already exists. Use ./run.sh or choose an empty APPWORLD_ROOT." >&2
      exit 2
    fi
    "$APPWORLD_PYTHON" -m pip install 'appworld @ git+https://github.com/stonybrooknlp/appworld.git@a072b7a86e7c1d5b1d7175659d750ebb9b79f10a'
    "$APPWORLD_PYTHON" -m appworld.cli install
    "$APPWORLD_PYTHON" -m appworld.cli download data --root "$APPWORLD_ROOT"
    [[ -s "$APPWORLD_ROOT/data/datasets/train.txt" ]] || { echo "Data download incomplete." >&2; exit 1; }
    ;;
  run|train)
    if [[ ! -s "${APPWORLD_ROOT:?Set APPWORLD_ROOT}/data/datasets/train.txt" ]]; then
      echo "AppWorld data missing. Run ./run.sh setup first." >&2; exit 2
    fi
    if [[ ${1:-run} == train ]]; then
      export PYTHONPATH="$PWD/verl:$PYTHONPATH"
      exec "${TRAIN_PYTHON:-python}" scripts/train.py
    fi
    exec "$APPWORLD_PYTHON" scripts/run_tasks.py
    ;;
esac
