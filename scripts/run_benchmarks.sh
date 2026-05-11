#!/usr/bin/env bash
# Sweep the experimental grid into a single CSV.
# Defaults to a small grid; pass --full for the full one.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="$ROOT/build/release/benchmark"
[[ -x "$BENCH" ]] || { echo "Build first: cmake --preset release && cmake --build --preset release" >&2; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/results/run_${TS}.csv"
mkdir -p "$(dirname "$OUT")"

REPS=(linear blocked roaring)
DENSITIES=(0.01 0.1 0.5)
TARGETS=(1 5 20)
SEEDS=(1 2 3)
N=10000
DIM=16
UNIVERSE=100000
QUERIES=500
WARMUP=50
EFS=(100)

if [[ "${1:-}" == "--full" ]]; then
    DENSITIES=(0.001 0.01 0.1 0.5 0.9)
    TARGETS=(1 5 20 100)
    EFS=(50 100 200)
    QUERIES=1000
fi

echo "Writing $OUT"
for seed in "${SEEDS[@]}"; do
    for rep in "${REPS[@]}"; do
        for dens in "${DENSITIES[@]}"; do
            for tgt in "${TARGETS[@]}"; do
                for ef in "${EFS[@]}"; do
                    "$BENCH" \
                        --representation "$rep" \
                        --n "$N" --dim "$DIM" --universe "$UNIVERSE" \
                        --density "$dens" --target-size "$tgt" \
                        --queries "$QUERIES" --warmup "$WARMUP" \
                        --ef "$ef" --seed "$seed" \
                        --out "$OUT"
                done
            done
        done
    done
done
echo "Done: $OUT"
