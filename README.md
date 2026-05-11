# Attribute-Filtered ANN Search - Representation Benchmark

A small, focused benchmark of attribute-set representations used inside the
filter callback of an HNSW approximate-nearest-neighbour search.

**What this project measures.** When each indexed point carries a large, sparse
set of attributes (e.g. tags) and a query asks for the nearest neighbours
*whose attributes contain a given target subset*, the filter test sits on
the hot path of HNSW traversal. This project compares three representations of
that attribute set on time, memory, and recall:

| Representation  | Header                                          | Notes                                            |
|-----------------|-------------------------------------------------|--------------------------------------------------|
| `linear`        | `include/anf/attributes/linear_bitset.hpp`      | Naive byte-array, linear-scan baseline           |
| `blocked`       | `include/anf/attributes/blocked_bitset.hpp`     | `uint64_t` words, short-circuit `b & ~a`         |
| `roaring`       | `include/anf/attributes/roaring_set.hpp`        | CRoaring compressed bitmap with `isSubset`        |

**What this project does *not* do.** It does not modify HNSW traversal - the
vendored `hnswlib` is upstream-pristine. There is no dynamic brute-force/ANN
switch.

## Build

Requires CMake ≥ 3.20 and a C++20 compiler.

```bash
cmake --preset release
cmake --build --preset release -j
ctest --preset release
```

Other presets: `debug`, `asan`.

## Refreshing third-party deps

```bash
scripts/install_deps.sh                      # default tags (hnswlib v0.8.0, CRoaring v4.2.1)
HNSWLIB_TAG=master scripts/install_deps.sh   # override
```

## Running the benchmark

A single binary, parameterised via CLI, appending one CSV row per run:

```bash
./build/release/benchmark \
    --representation roaring \
    --n 10000 --dim 16 --universe 100000 \
    --density 0.1 --target-size 5 \
    --queries 1000 --warmup 50 \
    --ef 100 --seed 42 \
    --out results/run.csv
```

Sweep the grid:

```bash
scripts/run_benchmarks.sh           # small grid
scripts/run_benchmarks.sh --full    # full grid
scripts/plot_results.py results/run_*.csv --out plots/
```

## Methodology, in one paragraph

For each cell of (representation × density × target_size × ef × seed) the
benchmark builds the HNSW index once, generates `--queries` query vectors and
targets, computes the exact filtered top-k by brute force as ground truth,
discards `--warmup` queries, then measures (a) filtered query latency and
recall@k against ground truth, (b) a no-filter baseline pass, and (c) a
filter-only microbenchmark that calls the predicate `n × queries` times to
isolate the cost of the bit operation from HNSW traversal. Selectivity (the
fraction of items passing the predicate) is recorded so latency can be
interpreted in context.

## License

MIT - see [LICENSE](LICENSE). Vendored deps keep their own licenses under `third_party/`.
