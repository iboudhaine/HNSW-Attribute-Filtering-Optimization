#!/usr/bin/env python3
"""Plot benchmark results.

Usage: plot_results.py results/run_*.csv [--out plots/]

Generates one PNG per (target_size, ef): latency p50 vs density per representation,
and a second figure with filter-only ns/call vs density.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["representation", "n", "dim", "density", "target_size", "ef"]
    return df.groupby(keys, as_index=False).agg(
        lat_p50=("lat_ns_p50", "mean"),
        lat_mean=("lat_ns_mean", "mean"),
        filter_only=("filter_only_ns_per_call", "mean"),
        recall=("recall_at_k", "mean"),
        selectivity=("selectivity", "mean"),
        bytes_per_item=("bytes_per_item", "mean"),
    )


def plot_panel(agg: pd.DataFrame, y: str, ylabel: str, title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for rep, sub in agg.groupby("representation"):
        sub = sub.sort_values("density")
        ax.plot(sub["density"], sub[y], marker="o", label=rep)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("item attribute density")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="+")
    p.add_argument("--out", default="plots")
    args = p.parse_args()

    df = pd.concat([pd.read_csv(c) for c in args.csv], ignore_index=True)
    agg = aggregate(df)
    os.makedirs(args.out, exist_ok=True)

    for (tgt, ef), sub in agg.groupby(["target_size", "ef"]):
        plot_panel(
            sub,
            "lat_p50",
            ylabel="filtered query p50 latency (ns)",
            title=f"target_size={tgt}, ef={ef}",
            path=os.path.join(args.out, f"latency_t{tgt}_ef{ef}.png"),
        )
        plot_panel(
            sub,
            "filter_only",
            ylabel="filter-only ns / call",
            title=f"filter-only cost  (target_size={tgt}, ef={ef})",
            path=os.path.join(args.out, f"filter_only_t{tgt}_ef{ef}.png"),
        )
    print(f"wrote plots to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
