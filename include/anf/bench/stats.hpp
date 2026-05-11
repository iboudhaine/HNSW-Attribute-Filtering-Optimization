#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace anf {

struct LatencyStats {
    double mean_ns = 0;
    double stdev_ns = 0;
    uint64_t p50_ns = 0;
    uint64_t p95_ns = 0;
    uint64_t p99_ns = 0;
};

// Mutates `samples` (sorts it). Cheap and avoids a copy.
inline LatencyStats summarise(std::vector<uint64_t>& samples) {
    LatencyStats s;
    if (samples.empty()) return s;

    std::sort(samples.begin(), samples.end());

    double sum = 0.0;
    for (uint64_t x : samples) sum += static_cast<double>(x);
    s.mean_ns = sum / static_cast<double>(samples.size());

    double sq = 0.0;
    for (uint64_t x : samples) {
        const double d = static_cast<double>(x) - s.mean_ns;
        sq += d * d;
    }
    s.stdev_ns = std::sqrt(sq / static_cast<double>(samples.size()));

    auto pct = [&](double p) {
        const size_t idx = std::min(samples.size() - 1,
                                    static_cast<size_t>(p * static_cast<double>(samples.size())));
        return samples[idx];
    };
    s.p50_ns = pct(0.50);
    s.p95_ns = pct(0.95);
    s.p99_ns = pct(0.99);
    return s;
}

}  // namespace anf
