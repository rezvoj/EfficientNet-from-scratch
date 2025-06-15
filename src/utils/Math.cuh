#pragma once


template <typename Integer>
__forceinline__ __host__ __device__
constexpr Integer ceilDiv(const Integer value, const Integer divisor) {
    return ((value + divisor - 1) / divisor);
}
