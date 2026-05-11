#!/usr/bin/env bash
# Refresh vendored third-party dependencies (hnswlib, CRoaring) into third_party/.
# Idempotent: re-running replaces the contents.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party"

HNSWLIB_TAG="${HNSWLIB_TAG:-v0.8.0}"
CROARING_TAG="${CROARING_TAG:-v4.2.1}"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Fetching hnswlib $HNSWLIB_TAG..."
git clone --depth 1 --branch "$HNSWLIB_TAG" \
    https://github.com/nmslib/hnswlib.git "$tmp/hnswlib"
rm -rf "$THIRD_PARTY/hnswlib"
mkdir -p "$THIRD_PARTY/hnswlib"
cp "$tmp/hnswlib/hnswlib/"*.h "$THIRD_PARTY/hnswlib/"
cp "$tmp/hnswlib/LICENSE" "$THIRD_PARTY/hnswlib/"
echo "$HNSWLIB_TAG" > "$THIRD_PARTY/hnswlib/VERSION"

echo "Fetching CRoaring $CROARING_TAG amalgamation..."
git clone --depth 1 --branch "$CROARING_TAG" \
    https://github.com/RoaringBitmap/CRoaring.git "$tmp/croaring"
( cd "$tmp/croaring" && ./amalgamation.sh >/dev/null )
rm -rf "$THIRD_PARTY/croaring"
mkdir -p "$THIRD_PARTY/croaring"
cp "$tmp/croaring/roaring.c" "$tmp/croaring/roaring.h" "$tmp/croaring/roaring.hh" \
   "$THIRD_PARTY/croaring/"
cp "$tmp/croaring/LICENSE" "$THIRD_PARTY/croaring/"
echo "$CROARING_TAG" > "$THIRD_PARTY/croaring/VERSION"

echo "Done. Dependencies refreshed under $THIRD_PARTY."
