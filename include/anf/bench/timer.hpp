#pragma once

#include <chrono>
#include <cstdint>

namespace anf {

class Timer {
public:
    Timer() : start_(clock::now()) {}
    void reset() { start_ = clock::now(); }
    [[nodiscard]] uint64_t elapsed_ns() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - start_).count();
    }

private:
    using clock = std::chrono::steady_clock;
    clock::time_point start_;
};

}  // namespace anf
