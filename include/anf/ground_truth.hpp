#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <vector>

#include "anf/dataset.hpp"

namespace anf {

inline float l2_sq(const float* a, const float* b, size_t dim) {
    float s = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        const float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

// True top-k by exact filtered brute force. `target` must be sorted.
// Returns k labels sorted by ascending distance.
inline std::vector<uint32_t> exact_filtered_topk(const Dataset& d, const float* query,
                                                 const std::vector<uint32_t>& target, size_t k) {
    using Pair = std::pair<float, uint32_t>;  // (distance, label)
    std::priority_queue<Pair> heap;           // max-heap on distance

    for (size_t i = 0; i < d.n; ++i) {
        const auto& attrs = d.attributes[i];
        // target is subset of attrs: both sorted, do a linear merge.
        size_t ti = 0, ai = 0;
        while (ti < target.size() && ai < attrs.size()) {
            if (target[ti] == attrs[ai]) {
                ++ti;
                ++ai;
            } else if (target[ti] > attrs[ai]) {
                ++ai;
            } else {
                break;
            }
        }
        if (ti != target.size()) continue;

        const float dist = l2_sq(query, d.vectors.data() + i * d.dim, d.dim);
        if (heap.size() < k) {
            heap.emplace(dist, static_cast<uint32_t>(i));
        } else if (dist < heap.top().first) {
            heap.pop();
            heap.emplace(dist, static_cast<uint32_t>(i));
        }
    }

    std::vector<Pair> result;
    result.reserve(heap.size());
    while (!heap.empty()) {
        result.push_back(heap.top());
        heap.pop();
    }
    std::sort(result.begin(), result.end(),
              [](const Pair& x, const Pair& y) { return x.first < y.first; });

    std::vector<uint32_t> labels;
    labels.reserve(result.size());
    for (const auto& p : result) labels.push_back(p.second);
    return labels;
}

inline double recall_at_k(const std::vector<uint32_t>& got, const std::vector<uint32_t>& truth) {
    if (truth.empty()) return 1.0;
    size_t hits = 0;
    for (uint32_t t : truth) {
        if (std::find(got.begin(), got.end(), t) != got.end()) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(truth.size());
}

}  // namespace anf
