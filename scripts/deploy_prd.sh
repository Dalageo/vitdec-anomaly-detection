#!/bin/bash

set -euo pipefail

if ! git diff --quiet HEAD; then
  echo "Error: You have uncommitted changes." >&2
  echo "Please commit or stash them before running this script." >&2
  exit 1
fi

echo "➡️ Fetching all updates from remote..."
git fetch origin

COMMIT_MSG=$(git log -1 --pretty=%B origin/tst)

echo "➡️ Checking out 'prd' branch..."
git checkout prd

echo "➡️ Pulling latest 'prd' branch from remote..."
git pull origin prd

echo "➡️ Merging 'origin/tst' into 'prd'..."
git merge origin/tst -m "Merge tst into prd: $COMMIT_MSG"


echo "➡️ Pushing merged 'prd' branch to remote..."
git push origin prd

git checkout dev

echo "🎉 Successfully promoted 'tst' to 'prd'."