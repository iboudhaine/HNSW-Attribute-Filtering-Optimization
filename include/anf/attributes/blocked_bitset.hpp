#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace anf {

// Word-blocked bitset: dense bits packed into uint64_t words.
// Why not std::bitset: it requires the size at
// compile time. We need a runtime-sized universe.
class BlockedBitset {
public:
    BlockedBitset() = default;
    BlockedBitset(size_t universe, std::span<const uint32_t> indices)
        : universe_(universe), words_((universe + 63) / 64, 0) {
        for (uint32_t i : indices) { words_[i >> 6] |= (uint64_t{1} << (i & 63)); }
    }

    static BlockedBitset from_indices(size_t universe, std::span<const uint32_t> indices) {
        return BlockedBitset(universe, indices);
    }

    static bool contains_subset(const BlockedBitset& a, const BlockedBitset& b) {
        const size_t n = a.words_.size();
        for (size_t i = 0; i < n; ++i) {
            if (b.words_[i] & ~a.words_[i]) return false;
        }
        return true;
    }

    static size_t size_in_bytes(const BlockedBitset& a) {
        return a.words_.size() * sizeof(uint64_t);
    }

    static constexpr const char* name() { return "blocked"; }

private:
    size_t universe_ = 0;
    std::vector<uint64_t> words_;
};

}  // namespace anf
