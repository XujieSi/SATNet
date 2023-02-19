#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <xmmintrin.h>
#include <smmintrin.h>
#include <float.h>
#include <vector>

#include <omp.h>

#include "satnet.h"

#define saxpy mysaxpy
#define scopy myscopy
#define sscal mysscal
#define sdot  mysdot
#define snrm2 mysnrm2
#define szero myszero
#define saturate mysaturate

const double MEPS = 1e-24;

/* 
 * Helper functions
 */

void saxpy(float* y, float a, const float* x, int l) {
    for (int i = 0; i < l; ++i) {
        y[i] += a * x[i];
    }
}

/*
void saxpy(float *__restrict__ y, float a, const float *__restrict__ x, int l)
{
    y = (float*)__builtin_assume_aligned(y, 4*sizeof(float));
    x = (float*)__builtin_assume_aligned(x, 4*sizeof(float));
    __m128 const a_ = _mm_set1_ps(a);
    for(int i=0; i<l; i+=4, x+=4, y+=4){
        __m128 y_ = _mm_load_ps(y);
        __m128 x_ = _mm_load_ps(x);
        y_ = _mm_add_ps(_mm_mul_ps(a_, x_), y_);
        _mm_store_ps(y, y_);
    }
}
*/

void scopy(float *x, float *y, int l)
{
        memcpy(y, x, sizeof(*x)*(size_t)l);
}

float sdot (const float* x, const float* y, int l) {
    float res = 0.0;
    for (int i = 0; i < l; ++i) {
        res += x[i] * y[i];
    }
    return res;
}
/* 
float sdot(const float *__restrict__ x, const float *__restrict__ y, int l)
{
    x = (float*)__builtin_assume_aligned(x, 4*sizeof(float));
    y = (float*)__builtin_assume_aligned(y, 4*sizeof(float));
    __m128 s = _mm_set1_ps(0);
    for(int i=0; i<l; i+=4, x+=4, y+=4){
        __m128 x_ = _mm_load_ps(x);
        __m128 y_ = _mm_load_ps(y);
        __m128 t = _mm_dp_ps(x_, y_, 0xf1);
        s = _mm_add_ss(s, t);
    }
    float s_;
    _mm_store_ss(&s_, s);

    return s_;
}
*/

void sscal(float *x, float a, int l)
{
        int m = l-4;
        int i;
        for (i = 0; i < m; i += 5){
                x[i] *= a;
                x[i+1] *= a;
                x[i+2] *= a;
                x[i+3] *= a;
                x[i+4] *= a;
        }

        for ( ; i < l; i++)        /* clean-up loop */
                x[i] *= a;
}

float snrm2(const float *x, int l)
{
        float xx = sdot(x, x, l);
        return sqrt(xx);
}
void szero(float *v, int l)
{
    memset(v, 0, l*sizeof(*v));
}

void dbgout_z (const int32_t *index, const float *V, int k) {
    for (int i,i_=0; (i=index[i_]); i_++) {
        //float zi = V[i*k];
        float zi = sdot(V, V+i*k, k);
        printf("True-vec  V[%d]\n", i);
        for (int j = 0; j < k; ++j) {
            printf("%10f,%10f\n", V[j], V[i*k + j]);
        }
        printf("i=%d zi = %f\n", i, zi);
    }
}


void dbgout1D(const char* before, const float* A, int len, const char* end) {
    printf("%s", before);
    for (int i=0; i<len; i++){
        printf("%7.4f ", A[i]);
    }
    printf("%s", end);
}

void dbgout2D(const char* before, const float* A, int R, int C, const char* end) {
    printf("%s", before);
    for(int i = 0; i < R; ++i) {
        dbgout1D("", A + i * C, C);
    }
    printf("%s", end);
}


void mix_init(int32_t *perm, int n, int k, const int32_t *is_input, int32_t *index, const float *z, float *V)
{
    //rand_unit(V+0, k);
    // normalize truth-vector
    float s = sdot(V, V, k);
    s = 1/sqrtf(s);
    sscal(V, s, k);

    // printf("in mix_init:\n");
    for (int i=0; i<n; i++) {
        // printf("i=%d  is_input: %s\n", i, is_input[i] ? "Yes" : "No");
        if (is_input[i]) {
            //float Vi1 = V[i*k+1];
            //szero(V+i*k, k);
            //V[i*k] = -cos(z[i]*M_PI);
            //V[i*k+1] = copysign(sin(z[i]*M_PI), Vi1);
            if (i > 0) {
                // initialization with Equation (5)
                const float a = -cos(z[i]*M_PI);
                const float b = sin(z[i]*M_PI);
                std::vector<float> tv;
                for (int j = 0; j < k; ++j) {
                    float r = 0.0;
                    for (int d = 0; d < k; ++d) {
                        if (j == d) {
                            r += (1.0 - V[j] * V[d]) * V[i*k + d];
                        }
                        else {
                            r += -V[j] * V[d] * V[i*k + d];
                        }
                    }
                    r *= b;
                    r += a * V[j];
                    tv.push_back(r);
                }
                for (int j = 0; j < k; ++j) {
                    V[i*k + j] = tv[j];
                }
            }
            // printf("truth-vec:\n");
            // for(int j = 0; j< k; ++j) printf("%7.4f ", V[j]);
            // printf("\ncur-vec:\n");
            // for(int j = 0; j< k; ++j) printf("%7.4f ", V[i * k + j]);
            // printf("\n");
        } else {
            float s = sdot(V+i*k, V+i*k, k);
            s = 1/sqrtf(s);
            sscal(V+i*k, s, k);
        }
    }
    int i_=0, j=0;
    for (; i_<n-1; i_++) {
        int i = perm[i_]+1;
        if (!is_input[i]) index[j++] = i;
    }
    for (; j<n; j++) index[j] = 0;
}

float mix_kernel(int is_forward, float prox_lam,
        int m, int k, const int32_t *__restrict__ index, 
        const float *__restrict__ S, const float *__restrict__ dz, float *__restrict__ V, const float *__restrict__ Vproj, float *__restrict__ W, 
        float *__restrict__ gnrm, const float *__restrict__ Snrms, float *__restrict__ g)
{
    float delta = 0;
    printf("\n");
    for (int i, i_=0; (i=index[i_]); i_++) {
        //printf("mix_kernel, i_ = %d\n", i_);
        const float Sii = Snrms[i];
        const float *__restrict__  Si = S+i*m;

        printf("mix_kernel, i_=%d, i = %d, *Si = %f, Sii=%f\n", i, i_, *Si, Sii);


        // Equation (A.3)
        // val = Wk'Si - Sii Vik
        for (int kk=0; kk<k; kk++)
            g[kk] = sdot(Si, W+kk*m, m);

        saxpy(g, -Sii, V+i*k, k);


        float gnrmi;
        if (is_forward) { 
            // gnrm is calculated in the forward pass
            gnrmi = snrm2(g, k);

            // dbgout1D("g:\n", g, k);
            // printf("gnrmi: %f\n", gnrmi);

            sscal(g, -1, k);
        } else {  
            gnrmi = gnrm[i]+prox_lam;


            // BEGIN OF THE NEW IMPLEMENTATION
            //
            // Equation (D.2)        
            // need to compute  - P_o ( U_o S'_o s_o - norm(s_o) * u_o - dloss/dv_o)
            // at this point, (similar to Equation A.3)
            // g = U_o S'_o s_o - norm(s_o) * u_o 
            // goal: -P_o (g - dloss/dv_o)
            // dloss/dv_o = dz_i * V_true

            saxpy(g, -dz[i], Vproj, k); // g - dloss/dv_o

            // P_o = I_k - v_o * v'_o
            std::vector<float> tv(k,0.0); 
            for (int r = 0; r < k; ++r) {
                for (int c = 0; c < k; ++c) {
                    float p =  - Vproj[i*k + r] * Vproj[i*k + c];
                    if (r == c) p += 1.0;
                    tv[r] += p * g[c];
                }
            }
            for(int r =0; r < k; ++r) {
                g[r] = tv[r];
            }

            sscal(g, -1, k);
            // END OF NEW IMPLEMENTATION 
        
            

            // g = -(I-v_i v_i') (g+v_0 dz[i])
            // float c = sdot(Vproj+i*k, g, k) + dz[i] * Vproj[i*k];
            // sscal(g, -1, k);
            // saxpy(g, c, Vproj+i*k, k);
            // g[0] -= dz[i]; // why only update g[0] using dz[i] ???
        }
        
        // Equation (A.5) as well as (D.3)
        sscal(g, 1/gnrmi, k);

        // dbgout1D("after rescaling by dividing gnrmi, g:\n", g, k);
        // since g has been rescaled, should we reset gnrmi to 1.0 at this point??
        // No. At this point, g basically serves as a temporary variable holding value for Vi (to be updated soon)


        // the goal here is to make g still a unit vector
        // however, the `gnrmi` is from the previous forward pass
        // while the newly updated `g` may become way larger than `gnrmi`
        // which is the root cause of why U/Phi would get exploded in the backward mixing phase

        float t;
        for (int kk=0; kk<k; kk++)
            t = g[kk], g[kk] -= V[i*k+kk], V[i*k+kk] = t;

        for (int kk = 0; kk < k; ++kk) {
            if (V[i*k +kk] > 100 || V[i*k + kk] < -100) {
                printf("the value is too large!!\n");
            }
        }

        // W += (vi^new-vi^old) Si'
        for (int kk=0; kk<k; kk++)
            saxpy(W+kk*m, g[kk], Si, m);

        if (is_forward) {
            // Calc function decrease, i.e., gnorm * (vi_new - vi_old)^2
            const float dec =  gnrmi * sdot(g, g, k);
            delta += dec;
            gnrm[i] = gnrmi;
            printf("coordinate update on dimention %d, dec: %f, accumulated delta: %f, gnrmi: %f\n", i, dec, delta, gnrmi);
        }
    }

    // dbgout_z(index, V, k);
    return delta; // only useful for the forward pass
}

inline float saturate(float x)
{
    return x - (x<0)*x + (x>1)*(1-x);
}

// consider the \min unsat problem,
void mix_forward(int max_iter, float eps, int n, int m, int k, const int32_t *index, int32_t *niter, const float *S, float *z, float *V, float *W, float *gnrm, float *Snrms, float *cache)
{
    float delta;
    int iter = 0;
    for (; iter < max_iter; iter++) {
        delta = mix_kernel(1, 0, m, k, index, S, NULL, V, NULL, W, gnrm, Snrms, cache);
    
        if (iter && delta < eps) break;
        if (iter == 0) eps = delta*eps;
    }

    *niter = iter;

    for (int i,i_=0; (i=index[i_]); i_++) {
        //float zi = V[i*k];
        float zi = sdot(V, V+i*k, k);
        // printf("True-vec  V[%d]\n", i);
        // for (int j = 0; j < k; ++j) {
        //     printf("%10f,%10f\n", V[j], V[i*k + j]);
        // }
        // printf("zi = %f\n", zi);

        zi = saturate((zi+1)/2)*2-1;
        zi = saturate(1-acosf(zi)/M_PI);
        z[i] = zi;
    }
}

void mix_backward(float prox_lam, int n, int m, int k, int32_t *is_input, int32_t *index, int32_t *niter, const float *S, float *dS, float *z, float *dz, const float *V, float *U, float *W, float *Phi, float *gnrm, float *Snrms, float *cache)
{
    int invalid_flag=0;
    for (int i,i_=0; (i=index[i_]); i_++) {
        float zi = z[i];
        // Equation (8)
        if (zi > 0.95) {
            zi = 0.95;
        } 
        if (zi < 0.05) {
            zi = 0.05;
        }
        float dzi = dz[i]/M_PI/sin(zi*M_PI);

        if (dzi < -10 || dzi > 10) {
            printf("dzi is too large!! i=%d dzi=%f\n", i, dzi);
        }

        // to avoid some tricky issue when 0/0
        // printf("before, dz[%d]=%f, dzi=%f\n", i, dz[i], dzi);
        // if (dz[i] < 1e-10) dzi = 0;
        // printf("after, dz[%d]=%f, dzi=%f\n", i, dz[i], dzi);

        if (isnan(dzi) || isinf(dzi) || gnrm[i] < MEPS) {
            printf("gnrm[%d] = %f\n", i, gnrm[i]);
            invalid_flag = 1;
        }
        dz[i] = dzi;
    }
    if (invalid_flag) { szero(dz, n); return; }

    printf("before mix_kernel call \n");
    dbgout2D("U:\n", U, n, k);
    dbgout2D("V:\n", V, n, k);
    dbgout2D("Phi:\n", Phi, k, m);
    
    // solve P (S'S+D_z-D_sii)xI_k P U = -dz P v0
    for (int iter=0; iter<*niter; iter++) {
        mix_kernel(0, prox_lam, m, k, index, S, dz, U, V, Phi, gnrm, Snrms, cache);
        printf("mix_backward, iter=%d\n", iter);
        dbgout2D("U:\n", U, n, k);
        dbgout2D("V:\n", V, n, k);
        dbgout2D("Phi:\n", Phi, k, m);
    }

    // sanity check
    for (int ik=0; ik<n*k; ik++) {
        if (isnan(U[ik]) || isinf(U[ik])) invalid_flag = 1;
    }
    if (invalid_flag) { szero(dz, n); return; }

    // dS = U W + V Phi
    // Equation (B.10)
    // V, U shape: nxk
    // W  =  V' S, shape: kxm 
    // Phi = U' S, shape: kxm, (more strictly should be U'_o S_o)
    // so dS = U V'S + V U'S = (UV' + VU')S ??
    // not exactly, dS = U V'S + V U'_o S_o ??
    // what is the definition for U'_I then? simply 0?
    for (int i=0; i<n; i++) {
        for (int kk=0; kk<k; kk++) {
            saxpy(dS+i*m, U[i*k+kk], W+kk*m, m);
            saxpy(dS+i*m, V[i*k+kk], Phi+kk*m, m);
        }
    }

    printf("after mix_kernel call \n");
    dbgout2D("U:\n", U, n, k);
    dbgout2D("V:\n", V, n, k);
    dbgout2D("Phi:\n", Phi, k, m);

    dbgout2D("dS after update:\n", dS, n, m);
    for(int i = 0; i < n; ++i){
        for(int j=0; j < m; ++j){
            if (dS[i*m + j] > 20) {
                printf("dS is too large!");
            }
        }
    }

    // dzi = v0'Phi si
    for (int i=1; i<n; i++) {
        if (!is_input[i]) {
             dz[i] = 0;
             continue;
        }
        float val1 = sdot(S+i*m, Phi+0*m, m), val2 = sdot(S+i*m, Phi+1*m, m); 
        dz[i] = (dz[i] + val1) * sin(z[i]*M_PI)*M_PI + val2 * copysign(cos(z[i]*M_PI)*M_PI, V[i*k+1])*M_PI;
    }
}

void mix_init_launcher_cpu(mix_t mix, int32_t *perm)
{
    int n=mix.n, k=mix.k; 
    #pragma omp parallel for schedule(dynamic)
    for (int i=0; i<mix.b; i++) {
        mix_init(perm,
            mix.n, mix.k, mix.is_input+i*n, mix.index+i*n, mix.z+i*n,
            mix.V+i*n*k);
    }
}

void mix_forward_launcher_cpu(mix_t mix, int max_iter, float eps)
{
    int n=mix.n, m=mix.m, k=mix.k;
    #pragma omp parallel for schedule(dynamic)
    for (int i=0; i<mix.b; i++) {
        printf("\n***batch = %d, mix_forward ***\n", i);
        // dbgout2D("before mixing, V:\n", mix.V + i * n * k, n, k);

        mix_forward(max_iter, eps,
            mix.n, mix.m, mix.k, mix.index+i*n, mix.niter+i, 
            mix.S, mix.z+i*n, mix.V+i*n*k, mix.W+i*m*k, mix.gnrm+i*n, mix.Snrms, mix.cache+i*k);
        
        // dbgout2D("after mixing, V:\n", mix.V + i * n * k, n, k);
    }
}

void mix_backward_launcher_cpu(mix_t mix, float prox_lam)
{
    int n=mix.n, m=mix.m, k=mix.k;
        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<mix.b; i++) {
            printf("\n***batch = %d, mix_backward ***\n", i);
            mix_backward(prox_lam,
               mix.n, mix.m, mix.k, mix.is_input+i*n, mix.index+i*n, mix.niter+i, 
               mix.S, mix.dS+i*n*m, mix.z+i*n, mix.dz+i*n, mix.V+i*n*k, mix.U+i*n*k, 
               mix.W+i*m*k, mix.Phi+i*m*k, mix.gnrm+i*n, mix.Snrms, mix.cache+i*k);
        }
}
