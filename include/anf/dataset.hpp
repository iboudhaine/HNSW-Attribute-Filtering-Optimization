#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace anf {

struct Dataset {
    size_t n = 0;
    size_t dim = 0;
    size_t universe = 0;
    std::vector<float> vectors;                     // n x dim, row-major
    std::vector<std::vector<uint32_t>> attributes;  // sorted indices per item
};

// Fisher-Yates: take the first k of a shuffled [0..universe).
// Linear in universe - fine here, and avoids the rejection-sampling pathology
// at high density.
inline std::vector<uint32_t> sample_indices(std::mt19937_64& rng, size_t universe, size_t k) {
    std::vector<uint32_t> pool(universe);
    std::iota(pool.begin(), pool.end(), 0u);
    for (size_t i = 0; i < k; ++i) {
        std::uniform_int_distribution<size_t> d(i, universe - 1);
        std::swap(pool[i], pool[d(rng)]);
    }
    pool.resize(k);
    std::sort(pool.begin(), pool.end());
    return pool;
}

inline Dataset make_synthetic(size_t n, size_t dim, size_t universe, double density,
                              uint64_t seed) {
    Dataset d;
    d.n = n;
    d.dim = dim;
    d.universe = universe;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    d.vectors.resize(n * dim);
    for (size_t i = 0; i < n * dim; ++i) d.vectors[i] = uniform(rng);

    const size_t k = static_cast<size_t>(static_cast<double>(universe) * density);
    d.attributes.resize(n);
    for (size_t i = 0; i < n; ++i) { d.attributes[i] = sample_indices(rng, universe, k); }
    return d;
}

inline std::vector<float> make_query_vectors(size_t n_queries, size_t dim, uint64_t seed) {
    std::vector<float> q(n_queries * dim);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    for (auto& x : q) x = uniform(rng);
    return q;
}

inline std::vector<std::vector<uint32_t>> make_targets(size_t n_queries, size_t universe,
                                                       size_t target_size, uint64_t seed) {
    std::vector<std::vector<uint32_t>> out(n_queries);
    std::mt19937_64 rng(seed);
    for (size_t i = 0; i < n_queries; ++i) { out[i] = sample_indices(rng, universe, target_size); }
    return out;
}

}  // namespace anf
