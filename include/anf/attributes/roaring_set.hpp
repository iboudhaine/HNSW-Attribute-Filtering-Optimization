#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "roaring.hh"

namespace anf {

// Compressed sparse bitmap.
// Designed to win when (density x universe) is small.
class RoaringSet {
public:
    RoaringSet() = default;
    RoaringSet(size_t /*universe*/, std::span<const uint32_t> indices) {
        bitmap_.addMany(indices.size(), indices.data());
        bitmap_.runOptimize();
        bitmap_.shrinkToFit();
    }

    static RoaringSet from_indices(size_t universe, std::span<const uint32_t> indices) {
        return RoaringSet(universe, indices);
    }

    static bool contains_subset(const RoaringSet& a, const RoaringSet& b) {
        return b.bitmap_.isSubset(a.bitmap_);
    }

    static size_t size_in_bytes(const RoaringSet& a) { return a.bitmap_.getSizeInBytes(); }

    static constexpr const char* name() { return "roaring"; }

private:
    roaring::Roaring bitmap_;
};

}  // namespace anf
