#!/usr/bin/env bash
# Deploy the latest commit to a Hugging Face Space.
#
#   ./deploy/push_to_hf.sh <hf-username>/<space-name>
#
# Pushes a fresh one-commit snapshot of HEAD (tracked files only, so .env is never
# included). PDFs are left out because Spaces reject binary files in plain git pushes.
set -euo pipefail

SPACE="${1:?Usage: $0 <hf-username>/<space-name>}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if [ -n "$(git status --porcelain)" ]; then
  echo "You have uncommitted changes. Commit them first so they are deployed." >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git archive HEAD | tar -x -C "$TMP"
find "$TMP" -name '*.pdf' -delete
cp deploy/huggingface/Dockerfile "$TMP/Dockerfile"
cat deploy/huggingface/space_header.md README.md > "$TMP/README.md"

COMMIT="$(git rev-parse --short HEAD)"
cd "$TMP"
git init -q -b main
git add -A
git -c user.name="$(git -C "$ROOT" config user.name)" \
    -c user.email="$(git -C "$ROOT" config user.email)" \
    commit -q -m "Deploy $COMMIT"
git push --force "https://huggingface.co/spaces/$SPACE" main

echo "Pushed. Watch the build at https://huggingface.co/spaces/$SPACE"
