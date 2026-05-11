#pragma once

#include <vector>

#include "hnswlib/hnswlib.h"

namespace anf {

// Generic subset filter: returns true iff target is a subset of attrs[label_id].
// Storage (attrs) is a contiguous vector indexed by internal label, so each
// filter call is one array access plus the representation's bit op.
template <class R>
class SubsetFilter final : public hnswlib::BaseFilterFunctor {
public:
    SubsetFilter(const std::vector<R>& attrs, const R& target) : attrs_(attrs), target_(target) {}

    bool operator()(hnswlib::labeltype label_id) override {
        return R::contains_subset(attrs_[label_id], target_);
    }

private:
    const std::vector<R>& attrs_;
    const R& target_;
};

}  // namespace anf
