#include <algorithm>

#include "anf/dataset.hpp"
#include "anf/ground_truth.hpp"
#include "doctest.h"

TEST_CASE("exact_filtered_topk returns only items containing the target") {
    auto d = anf::make_synthetic(/*n=*/500, /*dim=*/8,
                                 /*universe=*/256, /*density=*/0.25,
                                 /*seed=*/3);
    std::mt19937_64 rng(11);
    auto target = anf::sample_indices(rng, d.universe, 4);
    auto query = anf::make_query_vectors(1, d.dim, 17);

    auto top = anf::exact_filtered_topk(d, query.data(), target, 10);
    for (uint32_t lab : top) {
        const auto& attrs = d.attributes[lab];
        for (uint32_t t : target) {
            CHECK(std::find(attrs.begin(), attrs.end(), t) != attrs.end());
        }
    }
}

TEST_CASE("recall_at_k handles full hit and full miss") {
    std::vector<uint32_t> truth = {1, 2, 3, 4};
    CHECK(anf::recall_at_k({1, 2, 3, 4}, truth) == doctest::Approx(1.0));
    CHECK(anf::recall_at_k({9, 8, 7, 6}, truth) == doctest::Approx(0.0));
    CHECK(anf::recall_at_k({1, 2, 9, 9}, truth) == doctest::Approx(0.5));
}
