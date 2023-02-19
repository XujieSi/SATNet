#include <cstdint>
#include <cstdio>

typedef struct mix_t {
    int b, n, m, k;
    int32_t *is_input;  // b*n
    int32_t *index;     // b*n
    int32_t *niter;     // b
    float *S, *dS_sum;  // n*m
    float *dS;          // b*n*m
    float *z, *dz;      // b*n
    float *V, *U;       // b*n*k
    float *W, *Phi;     // b*m*k
    float *gnrm;        // b*n
    float *Snrms;       // n
    float *cache;
} mix_t ;

void mix_init_launcher_cpu(mix_t mix, int32_t *perm);

void mix_forward_launcher_cpu(mix_t mix, int max_iter, float eps);

void mix_backward_launcher_cpu(mix_t mix, float prox_lam);

void dbgout1D(const char* before, const float* A, int len, const char* end = "\n");

void dbgout2D(const char* before, const float* A, int R, int C, const char* end = "\n");