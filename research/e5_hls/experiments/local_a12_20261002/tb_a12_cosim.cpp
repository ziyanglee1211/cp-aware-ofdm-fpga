/* Same four fixed indices as rev-T; independent FFT PyTorch A12 golden. */
#include "weights.h"
#include "stim.h"
#include "stim_a12_preds.h"
#include <stdio.h>
#include <math.h>
void receiver_slot_opt(const float *, const float *, float *, float *);
int main() {
    static float yre[FF*SS], yim[FF*SS], l0[DD], l1[DD];
    const int slots[] = {0,19,22,30};
    int total=0, invalid=0;
    for (int s=0; s<4; ++s) {
        int b=slots[s];
        for (int i=0; i<FF*SS; ++i) {
            yre[i]=STIM_Y_RE[b*FF*SS+i]; yim[i]=STIM_Y_IM[b*FF*SS+i];
        }
        receiver_slot_opt(yre,yim,l0,l1);
        int mismatches=0;
        for (int i=0; i<DD; ++i) {
            if (!isfinite(l0[i]) || !isfinite(l1[i])) ++invalid;
            mismatches += (l1[i]>l0[i]) != (int)A12_PRED[b*DD+i];
        }
        total+=mismatches;
        printf("A12 FFT-contract slot %d: %d/%d mismatches\n", b,mismatches,DD);
    }
    printf("A12 FFT-contract total %d/1280 invalid=%d %s\n",total,invalid,(total==0 && invalid==0)?"PASS":"FAIL");
    return total==0 && invalid==0 ? 0:1;
}
