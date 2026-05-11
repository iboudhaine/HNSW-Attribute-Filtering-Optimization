// Sanity check: build a small dataset, run each filter representation, and
// confirm every returned label actually satisfies the predicate. Exits non-zero
// on any violation.

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "anf/attributes/blocked_bitset.hpp"
#include "anf/attributes/linear_bitset.hpp"
#include "anf/attributes/roaring_set.hpp"
#include "anf/dataset.hpp"
#include "anf/filter.hpp"
#include "hnswlib/hnswlib.h"

namespace {

bool target_subset_of(const std::vector<uint32_t>& attrs, const std::vector<uint32_t>& target) {
    size_t ti = 0, ai = 0;
    while (ti < target.size() && ai < attrs.size()) {
        if (target[ti] == attrs[ai]) {
            ++ti;
            ++ai;
        } else if (target[ti] > attrs[ai]) {
            ++ai;
        } else return false;
    }
    return ti == target.size();
}

template <class R>
int run_one(const anf::Dataset& d, const std::vector<uint32_t>& target_idx) {
    hnswlib::L2Space space(d.dim);
    hnswlib::HierarchicalNSW<float> hnsw(&space, d.n, 16, 200);
    for (size_t i = 0; i < d.n; ++i) { hnsw.addPoint(d.vectors.data() + i * d.dim, i); }

    std::vector<R> attrs;
    attrs.reserve(d.n);
    for (const auto& a : d.attributes) attrs.push_back(R::from_indices(d.universe, a));
    R target = R::from_indices(d.universe, target_idx);

    anf::SubsetFilter<R> filt(attrs, target);

    int violations = 0;
    for (size_t q = 0; q < 32; ++q) {
        auto res = hnsw.searchKnnCloserFirst(d.vectors.data() + (q % d.n) * d.dim, 5, &filt);
        for (auto& p : res) {
            if (!target_subset_of(d.attributes[p.second], target_idx)) { ++violations; }
        }
    }
    std::cout << "  " << R::name() << ": violations=" << violations << "\n";
    return violations;
}

}  // namespace

int main() {
    auto d = anf::make_synthetic(/*n=*/2000, /*dim=*/16, /*universe=*/1024,
                                 /*density=*/0.2, /*seed=*/123);
    std::mt19937_64 rng(7);
    auto target = anf::sample_indices(rng, d.universe, 3);

    int total = 0;
    total += run_one<anf::LinearBitset>(d, target);
    total += run_one<anf::BlockedBitset>(d, target);
    total += run_one<anf::RoaringSet>(d, target);
    if (total != 0) {
        std::cerr << "FAIL: " << total << " predicate violations\n";
        return 1;
    }
    std::cout << "OK\n";
    return 0;
}
