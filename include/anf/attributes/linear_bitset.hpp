#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace anf {

// Baseline: a heap-allocated boolean buffer with linear-scan subset test.
// Intentionally naive - exists to establish the upper bound on cost.
class LinearBitset {
public:
    LinearBitset() = default;
    LinearBitset(size_t universe, std::span<const uint32_t> indices) : bits_(universe, 0) {
        for (uint32_t i : indices) bits_[i] = 1;
    }

    static LinearBitset from_indices(size_t universe, std::span<const uint32_t> indices) {
        return LinearBitset(universe, indices);
    }

    static bool contains_subset(const LinearBitset& a, const LinearBitset& b) {
        const size_t n = a.bits_.size();
        for (size_t i = 0; i < n; ++i) {
            if (b.bits_[i] && !a.bits_[i]) return false;
        }
        return true;
    }

    static size_t size_in_bytes(const LinearBitset& a) { return a.bits_.size() * sizeof(uint8_t); }

    static constexpr const char* name() { return "linear"; }

private:
    std::vector<uint8_t> bits_;
};

}  // namespace anf
