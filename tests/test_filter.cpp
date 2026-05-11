// Integration smoke: each representation, plugged into HNSW as the filter
// callback, must only return labels that actually satisfy the predicate.

#include <algorithm>
#include <random>

#include "anf/attributes/blocked_bitset.hpp"
#include "anf/attributes/linear_bitset.hpp"
#include "anf/attributes/roaring_set.hpp"
#include "anf/dataset.hpp"
#include "anf/filter.hpp"
#include "doctest.h"
#include "hnswlib/hnswlib.h"

namespace {

bool ref_contains_all(const std::vector<uint32_t>& attrs, const std::vector<uint32_t>& target) {
    for (uint32_t t : target) {
        if (std::find(attrs.begin(), attrs.end(), t) == attrs.end()) return false;
    }
    return true;
}

template <class R>
void check_filter() {
    auto d = anf::make_synthetic(/*n=*/500, /*dim=*/8,
                                 /*universe=*/256, /*density=*/0.3,
                                 /*seed=*/7);
    hnswlib::L2Space space(d.dim);
    hnswlib::HierarchicalNSW<float> hnsw(&space, d.n, 16, 200);
    for (size_t i = 0; i < d.n; ++i) hnsw.addPoint(d.vectors.data() + i * d.dim, i);

    std::vector<R> attrs;
    attrs.reserve(d.n);
    for (const auto& a : d.attributes) attrs.push_back(R::from_indices(d.universe, a));

    std::mt19937_64 rng(13);
    auto target_idx = anf::sample_indices(rng, d.universe, 3);
    R target = R::from_indices(d.universe, target_idx);

    anf::SubsetFilter<R> filt(attrs, target);
    for (size_t q = 0; q < 20; ++q) {
        auto res = hnsw.searchKnnCloserFirst(d.vectors.data() + q * d.dim, 5, &filt);
        for (auto& p : res) { CHECK(ref_contains_all(d.attributes[p.second], target_idx)); }
    }
}

}  // namespace

TEST_CASE("LinearBitset filter only returns matching labels") { check_filter<anf::LinearBitset>(); }
TEST_CASE("BlockedBitset filter only returns matching labels") {
    check_filter<anf::BlockedBitset>();
}
TEST_CASE("RoaringSet filter only returns matching labels") { check_filter<anf::RoaringSet>(); }
