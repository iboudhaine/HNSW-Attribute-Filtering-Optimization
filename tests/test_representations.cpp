// Property test: random subsets - all three representations must agree
// with a std::set reference on the subset predicate.

#include <algorithm>
#include <random>
#include <set>

#include "anf/attributes/blocked_bitset.hpp"
#include "anf/attributes/linear_bitset.hpp"
#include "anf/attributes/roaring_set.hpp"
#include "anf/dataset.hpp"
#include "doctest.h"

namespace {

bool ref_subset(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    std::set<uint32_t> sa(a.begin(), a.end());
    for (uint32_t x : b)
        if (!sa.count(x)) return false;
    return true;
}

template <class R>
bool rep_subset(size_t universe, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    return R::contains_subset(R::from_indices(universe, a), R::from_indices(universe, b));
}

}  // namespace

TEST_CASE("all representations agree with reference on random subsets") {
    constexpr size_t universe = 4096;
    std::mt19937_64 rng(99);

    for (int trial = 0; trial < 200; ++trial) {
        const size_t na = 1 + rng() % 200;
        const size_t nb = 1 + rng() % 50;
        auto a = anf::sample_indices(rng, universe, na);
        auto b = anf::sample_indices(rng, universe, nb);

        // Sometimes force b to be a subset of a so we exercise both branches.
        if (rng() % 3 == 0 && !a.empty()) {
            b.clear();
            std::sample(a.begin(), a.end(), std::back_inserter(b), std::min<size_t>(5, a.size()),
                        rng);
            std::sort(b.begin(), b.end());
        }

        const bool truth = ref_subset(a, b);
        CHECK(rep_subset<anf::LinearBitset>(universe, a, b) == truth);
        CHECK(rep_subset<anf::BlockedBitset>(universe, a, b) == truth);
        CHECK(rep_subset<anf::RoaringSet>(universe, a, b) == truth);
    }
}

TEST_CASE("empty target is always a subset") {
    const std::vector<uint32_t> a = {1, 5, 9};
    const std::vector<uint32_t> empty;
    CHECK(rep_subset<anf::LinearBitset>(16, a, empty));
    CHECK(rep_subset<anf::BlockedBitset>(16, a, empty));
    CHECK(rep_subset<anf::RoaringSet>(16, a, empty));
}
