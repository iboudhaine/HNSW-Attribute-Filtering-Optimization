// Single, parameterised filter-representation benchmark.
//
// Methodology (per run):
//   1. Build HNSW (once).
//   2. Pre-generate queries + targets.
//   3. Compute exact filtered top-k (ground truth) by brute force.
//   4. Warm-up: discard the first --warmup queries.
//   5. Measure remaining queries; record latency + recall.
//   6. Run a no-filter baseline pass.
//   7. Run a filter-only microbenchmark (representation cost in isolation).
//   8. Emit one CSV row to --out (append-mode; header if file is empty).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "anf/attributes/blocked_bitset.hpp"
#include "anf/attributes/linear_bitset.hpp"
#include "anf/attributes/roaring_set.hpp"
#include "anf/bench/stats.hpp"
#include "anf/bench/timer.hpp"
#include "anf/dataset.hpp"
#include "anf/filter.hpp"
#include "anf/ground_truth.hpp"
#include "hnswlib/hnswlib.h"

namespace {

struct Config {
    std::string representation = "blocked";
    size_t n = 10000;
    size_t dim = 16;
    size_t universe = 100000;
    double density = 0.1;
    size_t target_size = 5;
    size_t queries = 1000;
    size_t warmup = 50;
    size_t k = 10;
    size_t M = 16;
    size_t ef_construction = 200;
    size_t ef_search = 100;
    uint64_t seed = 42;
    std::string out = "results/run.csv";
};

void usage() {
    std::cerr << "Usage: benchmark [--representation linear|blocked|roaring]\n"
                 "                 [--n N] [--dim D] [--universe U] [--density f]\n"
                 "                 [--target-size T] [--queries Q] [--warmup W]\n"
                 "                 [--k K] [--ef E] [--ef-construction EC] [--M M]\n"
                 "                 [--seed S] [--out PATH]\n";
}

bool parse(int argc, char** argv, Config& c) {
    auto need = [&](int& i, const char* flag) {
        if (i + 1 >= argc) {
            std::cerr << "missing value for " << flag << "\n";
            return false;
        }
        return true;
    };
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--representation" && need(i, "--representation")) c.representation = argv[++i];
        else if (a == "--n" && need(i, "--n")) c.n = std::stoull(argv[++i]);
        else if (a == "--dim" && need(i, "--dim")) c.dim = std::stoull(argv[++i]);
        else if (a == "--universe" && need(i, "--universe")) c.universe = std::stoull(argv[++i]);
        else if (a == "--density" && need(i, "--density")) c.density = std::stod(argv[++i]);
        else if (a == "--target-size" && need(i, "--target-size"))
            c.target_size = std::stoull(argv[++i]);
        else if (a == "--queries" && need(i, "--queries")) c.queries = std::stoull(argv[++i]);
        else if (a == "--warmup" && need(i, "--warmup")) c.warmup = std::stoull(argv[++i]);
        else if (a == "--k" && need(i, "--k")) c.k = std::stoull(argv[++i]);
        else if (a == "--ef" && need(i, "--ef")) c.ef_search = std::stoull(argv[++i]);
        else if (a == "--ef-construction" && need(i, "--ef-construction"))
            c.ef_construction = std::stoull(argv[++i]);
        else if (a == "--M" && need(i, "--M")) c.M = std::stoull(argv[++i]);
        else if (a == "--seed" && need(i, "--seed")) c.seed = std::stoull(argv[++i]);
        else if (a == "--out" && need(i, "--out")) c.out = argv[++i];
        else if (a == "-h" || a == "--help") {
            usage();
            return false;
        } else {
            std::cerr << "unknown arg: " << a << "\n";
            usage();
            return false;
        }
    }
    return true;
}

template <class R>
std::vector<R> build_attrs(const anf::Dataset& d) {
    std::vector<R> attrs;
    attrs.reserve(d.n);
    for (const auto& idx : d.attributes) { attrs.push_back(R::from_indices(d.universe, idx)); }
    return attrs;
}

template <class R>
size_t total_bytes(const std::vector<R>& attrs) {
    size_t s = 0;
    for (const auto& a : attrs) s += R::size_in_bytes(a);
    return s;
}

struct RunResult {
    anf::LatencyStats lat;
    anf::LatencyStats lat_no_filter;
    double filter_only_ns_per_call = 0.0;
    double recall_mean = 0.0;
    double selectivity_mean = 0.0;
    size_t bytes_per_item = 0;
    size_t index_bytes_total = 0;
};

template <class R>
RunResult run_with(const anf::Dataset& d, const Config& c, const std::vector<float>& queries,
                   const std::vector<std::vector<uint32_t>>& targets) {
    // Build HNSW.
    hnswlib::L2Space space(d.dim);
    hnswlib::HierarchicalNSW<float> hnsw(&space, d.n, c.M, c.ef_construction);
    for (size_t i = 0; i < d.n; ++i) { hnsw.addPoint(d.vectors.data() + i * d.dim, i); }
    hnsw.setEf(c.ef_search);

    // Build representation-specific attribute store + targets.
    auto attrs = build_attrs<R>(d);
    std::vector<R> targets_r;
    targets_r.reserve(targets.size());
    for (const auto& t : targets) targets_r.push_back(R::from_indices(d.universe, t));

    RunResult rr;
    rr.bytes_per_item = attrs.empty() ? 0 : total_bytes(attrs) / attrs.size();
    rr.index_bytes_total = total_bytes(attrs);

    // Ground truth (exact filtered top-k, brute force).
    std::vector<std::vector<uint32_t>> truth(c.queries);
    for (size_t q = 0; q < c.queries; ++q) {
        truth[q] = anf::exact_filtered_topk(d, queries.data() + q * d.dim, targets[q], c.k);
    }

    // Selectivity (fraction of items passing the predicate), averaged.
    {
        double s_sum = 0.0;
        for (size_t q = 0; q < c.queries; ++q) {
            size_t pass = 0;
            for (size_t i = 0; i < d.n; ++i) {
                if (R::contains_subset(attrs[i], targets_r[q])) ++pass;
            }
            s_sum += static_cast<double>(pass) / static_cast<double>(d.n);
        }
        rr.selectivity_mean = s_sum / static_cast<double>(c.queries);
    }

    // Filter-only microbenchmark: time the predicate alone, no traversal.
    {
        size_t calls = 0;
        anf::Timer t;
        // Use the same NxQ calls so the measurement is over the same work.
        for (size_t q = 0; q < c.queries; ++q) {
            for (size_t i = 0; i < d.n; ++i) {
                volatile bool b = R::contains_subset(attrs[i], targets_r[q]);
                (void)b;
                ++calls;
            }
        }
        const uint64_t ns = t.elapsed_ns();
        rr.filter_only_ns_per_call = static_cast<double>(ns) / static_cast<double>(calls);
    }

    // Filtered query latency + recall.
    std::vector<uint64_t> samples;
    samples.reserve(c.queries);
    double recall_sum = 0.0;

    for (size_t q = 0; q < c.queries; ++q) {
        anf::SubsetFilter<R> filt(attrs, targets_r[q]);
        anf::Timer t;
        auto res = hnsw.searchKnnCloserFirst(queries.data() + q * d.dim, c.k, &filt);
        const uint64_t ns = t.elapsed_ns();
        if (q >= c.warmup) {
            samples.push_back(ns);
            std::vector<uint32_t> got;
            got.reserve(res.size());
            for (auto& p : res) got.push_back(static_cast<uint32_t>(p.second));
            recall_sum += anf::recall_at_k(got, truth[q]);
        }
    }
    rr.lat = anf::summarise(samples);
    const size_t measured = c.queries - std::min(c.warmup, c.queries);
    rr.recall_mean = measured ? recall_sum / static_cast<double>(measured) : 0.0;

    // No-filter baseline - same queries, no predicate.
    std::vector<uint64_t> base_samples;
    base_samples.reserve(c.queries);
    for (size_t q = 0; q < c.queries; ++q) {
        anf::Timer t;
        auto res = hnsw.searchKnnCloserFirst(queries.data() + q * d.dim, c.k);
        (void)res;
        const uint64_t ns = t.elapsed_ns();
        if (q >= c.warmup) base_samples.push_back(ns);
    }
    rr.lat_no_filter = anf::summarise(base_samples);

    return rr;
}

void write_csv(const Config& c, const RunResult& r) {
    namespace fs = std::filesystem;
    fs::path out(c.out);
    if (out.has_parent_path()) fs::create_directories(out.parent_path());
    const bool exists = fs::exists(out) && fs::file_size(out) > 0;

    std::ofstream f(out, std::ios::app);
    if (!exists) {
        f << "seed,representation,n,dim,universe,density,target_size,ef,k,"
             "queries,warmup,selectivity,recall_at_k,"
             "lat_ns_mean,lat_ns_stdev,lat_ns_p50,lat_ns_p95,lat_ns_p99,"
             "baseline_ns_mean,baseline_ns_p50,"
             "filter_only_ns_per_call,bytes_per_item,index_bytes\n";
    }
    f << c.seed << ',' << c.representation << ',' << c.n << ',' << c.dim << ',' << c.universe << ','
      << c.density << ',' << c.target_size << ',' << c.ef_search << ',' << c.k << ',' << c.queries
      << ',' << c.warmup << ',' << r.selectivity_mean << ',' << r.recall_mean << ','
      << r.lat.mean_ns << ',' << r.lat.stdev_ns << ',' << r.lat.p50_ns << ',' << r.lat.p95_ns << ','
      << r.lat.p99_ns << ',' << r.lat_no_filter.mean_ns << ',' << r.lat_no_filter.p50_ns << ','
      << r.filter_only_ns_per_call << ',' << r.bytes_per_item << ',' << r.index_bytes_total << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    Config c;
    if (!parse(argc, argv, c)) return 1;

    std::cerr << "[bench] rep=" << c.representation << " n=" << c.n << " dim=" << c.dim
              << " density=" << c.density << " target=" << c.target_size << " ef=" << c.ef_search
              << " queries=" << c.queries << " seed=" << c.seed << "\n";

    auto d = anf::make_synthetic(c.n, c.dim, c.universe, c.density, c.seed);
    auto queries = anf::make_query_vectors(c.queries, c.dim, c.seed ^ 0xC0FFEE);
    auto targets = anf::make_targets(c.queries, c.universe, c.target_size, c.seed ^ 0xBEEF);

    RunResult r;
    if (c.representation == "linear") {
        r = run_with<anf::LinearBitset>(d, c, queries, targets);
    } else if (c.representation == "blocked") {
        r = run_with<anf::BlockedBitset>(d, c, queries, targets);
    } else if (c.representation == "roaring") {
        r = run_with<anf::RoaringSet>(d, c, queries, targets);
    } else {
        std::cerr << "unknown representation: " << c.representation << "\n";
        return 1;
    }

    write_csv(c, r);

    std::cerr << "[bench] selectivity=" << r.selectivity_mean << " recall=" << r.recall_mean
              << " lat_p50_ns=" << r.lat.p50_ns << " filter_only_ns=" << r.filter_only_ns_per_call
              << " bytes/item=" << r.bytes_per_item << "\n";
    return 0;
}
