#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
if ! command -v bwrap >/dev/null 2>&1; then
  if [ "$(id -u)" -ne 0 ]; then
    printf 'Missing bubblewrap. Install the system package, then rerun setup.\n' >&2
    exit 1
  fi
  printf 'Installing missing bubblewrap for the existing CPU execution tool.\n'
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends bubblewrap
fi
if command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
else
  INSTALLER="$(mktemp)"
  trap 'rm -f -- "$INSTALLER"' EXIT
  curl --fail --location --silent --show-error \
    https://astral.sh/uv/0.11.31/install.sh --output "$INSTALLER"
  UV_INSTALL_DIR="$ROOT/.tools" UV_NO_MODIFY_PATH=1 sh "$INSTALLER"
  UV="$ROOT/.tools/uv"
fi
"$UV" sync --locked --extra inference
