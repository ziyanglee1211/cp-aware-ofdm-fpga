#include <stdint.h>
#include <math.h>
#include "local_a12_quant.h"
void receiver_slot_opt(const float *, const float *, float *, float *);
extern "C" __declspec(dllexport) int run_slots(int count, const float *yr,
    const float *yi, uint8_t *pred, float *margins) {
    float l0[320], l1[320];
    for (int b = 0; b < count; ++b) {
        receiver_slot_opt(yr + b*560, yi + b*560, l0, l1);
        for (int i = 0; i < 320; ++i) {
            if (!isfinite(l0[i]) || !isfinite(l1[i])) return b+1;
            pred[b*320+i] = l1[i] > l0[i];
            margins[b*320+i] = l1[i] - l0[i];
        }
    }
    return 0;
}
extern "C" __declspec(dllexport) void quantize12(int n, const float *x, int32_t *out) {
    for (int i=0; i<n; ++i) out[i] = q12(x[i]);
}
extern "C" __declspec(dllexport) float get_scale12(int imaginary) {
    return imaginary ? LOCAL_A12_IM : LOCAL_A12_RE;
}
