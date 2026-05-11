#include <set>

#include "anf/dataset.hpp"
#include "doctest.h"

TEST_CASE("sample_indices produces exactly k unique sorted values in range") {
    std::mt19937_64 rng(1);
    for (size_t k : {0u, 1u, 5u, 100u, 1000u}) {
        auto v = anf::sample_indices(rng, /*universe=*/1000, k);
        CHECK(v.size() == k);
        std::set<uint32_t> uniq(v.begin(), v.end());
        CHECK(uniq.size() == k);
        for (uint32_t x : v) CHECK(x < 1000);
        for (size_t i = 1; i < v.size(); ++i) CHECK(v[i - 1] < v[i]);
    }
}

TEST_CASE("make_synthetic respects density exactly") {
    auto d = anf::make_synthetic(/*n=*/100, /*dim=*/4,
                                 /*universe=*/1000, /*density=*/0.1,
                                 /*seed=*/42);
    CHECK(d.n == 100);
    CHECK(d.dim == 4);
    CHECK(d.vectors.size() == 100 * 4);
    CHECK(d.attributes.size() == 100);
    for (const auto& a : d.attributes) CHECK(a.size() == 100);  // 10% of 1000
}
