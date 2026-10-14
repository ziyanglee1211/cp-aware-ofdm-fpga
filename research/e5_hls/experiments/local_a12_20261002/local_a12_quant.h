#pragma once
#include <stdint.h>
// Derived in float32 from frozen seed-1 checkpoint maxima, not rounded A8 scales.
#define LOCAL_A12_RE 0.00286863232f
#define LOCAL_A12_IM 0.00320664607f
static inline int q12(float qs) {
#pragma HLS INLINE
    if (qs > 2047.f) qs = 2047.f;
    else if (qs < -2047.f) qs = -2047.f;
    union { float f; uint32_t u; } m;
    m.f = qs + 12582912.0f;
    return (int)(m.u & 0x7FFFFFu) - 4194304;
}
