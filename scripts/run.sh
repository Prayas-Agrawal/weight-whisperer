#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export PATH="$PWD/.tools:$PATH"

case "${1:-}" in
  setup) exec bash scripts/setup.sh ;;
  lock) exec uv lock ;;
  check)
    .venv/bin/python -m ruff check src tests
    exec .venv/bin/python -m pytest -q
    ;;
  format)
    shift
    if [ "$#" -eq 0 ]; then set -- src tests; fi
    .venv/bin/python -m ruff format "$@"
    exec .venv/bin/python -m ruff check --fix "$@"
    ;;
  prepare) shift; exec .venv/bin/python -m wsp.prepare "$@" ;;
  verify-result) exec .venv/bin/python -m wsp.verify ;;
  inspect-edit) shift; exec .venv/bin/python -m wsp.inspect_edit "$@" ;;
  check-runtime) shift; exec .venv/bin/python -m wsp.check_runtime "$@" ;;
  rebuild) shift; exec .venv/bin/python -m wsp.rebuild "$@" ;;
  serve-editor) exec .venv/bin/python -m wsp.services start ;;
  stop-editor) exec .venv/bin/python -m wsp.services stop ;;
  research) shift; exec .venv/bin/python -m wsp.editor "$@" ;;
  evaluate) shift; exec .venv/bin/python -m wsp.benchmark "$@" ;;
  status) shift; exec .venv/bin/python -m wsp.benchmark --status "$@" ;;
  reproduce)
    if [ "$#" -ne 2 ]; then printf 'Usage: bash scripts/run.sh reproduce RUN_NAME\n' >&2; exit 2; fi
    .venv/bin/python -m wsp.prepare --models target editor
    .venv/bin/python -m wsp.editor "$2"
    exec .venv/bin/python -m wsp.benchmark "$2" --run "$2"
    ;;
  *)
    printf '%s\n' \
      'Usage: bash scripts/run.sh COMMAND [ARGUMENTS]' \
      '  setup | prepare [--models target editor] [--cache DIRECTORY]' \
      '  verify-result | inspect-edit [--edit run1|run2] | check-runtime [--full]' \
      '  rebuild [--edit run1|run2 | --run NAME]' \
      '  research NAME | evaluate NAME [--edit run1|run2 | --run NAME] [--gpus 0 1 2 3]' \
      '  status NAME | reproduce NAME | serve-editor | stop-editor' \
      '  check | format [FILES...] | lock' >&2
    exit 2
    ;;
esac
