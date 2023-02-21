

#include "satnet.h"
#include <vector>
#include <cstring>
#include <iostream>
#include <random>
#include <cstdalign>

std::default_random_engine generator;
std::normal_distribution<float> distribution(0.0, 1.0);

struct AdamSNet {
    const float eps = 1e-8;
    float alpha = 0.01;   // step size
    float *m, beta1 = 0.9, beta1_t=1.0; // 1st order moment and decay
    float *v, beta2 = 0.999, beta2_t=1.0; // 2nd order moment and decay
    int rows, cols;

    AdamSNet(int r, int c) : rows(r), cols(c) {
        m = new float [r * c];
        v = new float [r * c];
        memset(m, 0, sizeof(m));
        memset(v, 0, sizeof(v));
    }

    void step(float *theta, const float *grad) {
        beta1_t *= beta1;
        beta2_t *= beta2;

        // skip any update for the truth vector (i.e., 0-th row)
        for (int r = 1; r < rows; ++r) {
            for(int c = 0; c < cols; ++c) {
                const int index = r * cols + c;
                const float g = grad[index];
                const float g2 = g * g;

                m[index] = beta1 * m[index] + (1.0 - beta1) * g;
                v[index] = beta2 * v[index] + (1.0 - beta2) * g2;

                const float c_m = m[index] / (1.0 - beta1_t);
                const float c_v = v[index] / (1.0 - beta2_t);

                theta[index] -= alpha * c_m / (eps + fsqrt(c_v)); // our grads have been negated before this
            }
        }

    }
    
};

struct Learner
{
    int nvar, mclauses, aux;

    std::vector<std::vector<bool>> assignments; 
    std::vector<int> input_dims; 
    std::vector<bool> io_dims;
    std::vector<int> perm;

    Learner(std::vector<int>&& ins, std::vector<std::vector<bool>>&& assg, int m, int a = 0) : input_dims(ins), assignments(assg), mclauses(m), aux(a) {
        const int n = assignments[0].size() + a + 1; // one extra variable for the truth vector
        nvar = n;
        perm.clear();
        io_dims.clear();
        for (int i = 0; i < n; ++i) {
            perm.push_back(i);
            io_dims.push_back(false);
        }
        io_dims[0] = true;
        for (auto x : input_dims) {
            io_dims[x] = true;
        }
    }

    void construct_mix_params(mix_t &params)
    {
        const int n = nvar;
        const int m = mclauses;
        const int k = 16; // embedding dimension size
        const int b = assignments.size(); // batch
        params.n = n;
        params.m = m;
        params.b = b;
        params.k = k;

        // input/output learning signals
        params.is_input = new int32_t[b * n];
        memset(params.is_input, 0, sizeof(params.is_input));
        params.z = new float[b * n];

        for (int j = 0; j < b; ++j) {
            params.z[j * n] = 1.0;
            for (int i = 1; i < n; ++i){
                if (i <= assignments[j].size()) {
                    params.z[j * n + i] = assignments[j].at(i-1) ? 1.0 : 0.0;
                }
                else {
                    // assignment for auxiliary variables
                    params.z[j * n + i] = 0.5;
                }
            }
            params.is_input[j * n] = 1;
            for (auto x : input_dims) {
                params.is_input[j * n + x] = 1;
            }
        }
        params.dz = new float[b * n];

        // mixing method update schedule
        params.index = new int32_t[b * n];

        params.niter = new int32_t[b]; 
        for (int i = 0; i < b; ++i) {
            params.niter[i] = 40;
        }

        params.S = new float[n * m];
        float r = 0.5 / fsqrt(n);
        for (int i = 0; i < n * m; ++i) {
            // truth-vector
            if (i < m){
                params.S[i] = -r;
                continue;
            }

            params.S[i] = r * distribution(generator);
        }

        params.dS_sum = new float [n*m];
        params.dS = new float[ b * n * m];
        params.Snrms = new float[n];
        compute_Snrms(params);


        params.W = new float[b * k * m];
        params.Phi = new float[b * k * m];

        // memset(params.cache, 0, sizeof(params.cache));
        // memset(params.gnrm, 0, sizeof(params.gnrm));

        params.V = new float [b * n * k];
        for (int i = 0; i < b * n * k; ++i) {
            params.V[i] = distribution(generator);
        }
        params.U = new float [b * n * k];

        params.cache = new float[b * k];
        params.gnrm = new float[b * n];
        // for (int i = 0; i < n; ++i)
        // {
        //     params.gnrm[i] = distribution(generator);
        // }
    }

    void compute_Snrms(mix_t& params){
        const int n = params.n;
        const int m = params.m;
        for (int i = 0; i < n; ++i) {
            float r = 0.0;
            for (int j = 0; j < m; ++j) {
                const float sij = params.S[i * m + j];
                r += sij * sij;
            }
            params.Snrms[i] = r;
        }
    }

    void compute_dz(mix_t& params ) {
        memset(params.dz, 0, sizeof(params.dz));

        const int n = params.n;
        for (int j = 0; j < params.b; ++j){
            for (int i = 0; i < n; ++i) {
                if(io_dims[i]) continue; 
                const int index = j * n + i;
                if (i <= assignments[j].size()){
                    params.dz[index] = 2 * params.z[index] - 2.0 * (assignments[j].at(i-1) ? 1.0 : 0.0); 
                }

                printf("batch = %d, dloss/dv[%d] = %.6f ", j, i, params.dz[index]);
            }
            printf("\n");
        }
    }

    void compute_loss(mix_t& params ) {

        float loss = 0.0;
        const int n = params.n;
        for (int j = 0; j < params.b; ++j){
            float b_loss = 0.0;
            for (int i = 0; i < n; ++i) {
                if(io_dims[i]) continue; 
                const int index = j * n + i;
                if (i <= assignments[j].size()){
                    float groundtruth = (assignments[j].at(i-1) ? 1.0 : 0.0);
                    float delta = (params.z[index] - groundtruth);
                    b_loss -=  delta * delta;
                }
            }
            printf("batch=%d, b_loss=%f\n", j, b_loss);
            loss += b_loss;
        }
        printf("total loss: %f\n", loss);
    }

    void prepare_W(mix_t& params){
        const int n = params.n;
        const int m = params.m;
        const int k = params.k;
        const int b = params.b;

        // W = V^T * S
        // should be perfomed AFTER V is normalized
        for (int _b = 0; _b < b; ++ _b) {
            //printf("\n*** computing W, batch = %d ***\n", _b);
            const int base = _b * k * m;
            for (int i = 0; i < k; ++i) {
                for  (int j = 0; j < m; ++j) {
                    float res = 0.0; 
                    for (int d = 0; d < n; ++d) {
                        res += params.V[base + d *k + i] * params.S[d * m + j];
                    }
                    int index = i * m  + j;
                    params.W[base + index] = res;
                }
            }

        }
    }

    void show_dS(mix_t& params) {
        const int n = params.n;
        const int m = params.m;
        const int b = params.b; 
        for (int _b = 0 ; _b < b; ++ _b) {
            printf("batch=%d, dS:\n", _b);            
            dbgout2D("", params.dS + _b * n * m, n, m);
        }
    }

    void compute_dS_sum(mix_t& params) {
        const int n = params.n;
        const int m = params.m;
        const int b = params.b; 
        for (int i = 0; i < n; ++i){
            for (int j = 0; j < m; ++j) {
                float r = 0.0;
                for (int _b = 0 ; _b < b; ++ _b) {
                    const int base = _b * n * m; 
                    r += params.dS[base + i * m + j];
                }
                params.dS_sum[i * m + j] = r;
            }
        }
    }

    void update_S(mix_t& params) {
        const int n = params.n;
        const int m = params.m;
        // Note: skip any update for the truth-vector
        for (int i = 1; i < n; ++i){
            for (int j = 0; j < m; ++j) {
                const int index = i * m + j;
                params.S[index] +=  -params.dS_sum[index];
            }
        }
    }

    void forward(mix_t& params){
        mix_init_launcher_cpu(params, perm.data());
        prepare_W(params);

        // dbgout2D("in forward, before mixing, S:\n", params.S, params.n, params.m);

        mix_forward_launcher_cpu(params, 10, 1e-10);

        printf("after mixing, V:");
        for (int i=0; i< params.b; ++i) {
            printf("for batch = %d\n", i);
            const int base = i * params.n * params.k; 
            dbgout2D("", params.V + base, params.n, params.k);
        }

        dbgout2D("\npredicted outputs after forward process:\n", params.z, params.b, params.n);
        dbgout2D("in forward, S:\n", params.S, params.n, params.m);

        dbgout2D("gnrm:\n", params.gnrm, params.b, params.n);
        
        int dbg = 0;

    }

    void backward(mix_t& params) {
        // dz depends on the downstream loss function
        compute_loss(params);
        printf("\n");
        compute_dz(params);

        memset(params.dS, 0, sizeof(params.dS));
        memset(params.Phi, 0, sizeof(params.Phi));
        memset(params.U, 0, sizeof(params.U));

        float prox_lam = 0.01;

        //show_dS(params);

        // compute gradient using Mixing method
        mix_backward_launcher_cpu(params, prox_lam);



        // update S
        // update_S(params);
        // dbgout2D("updated S:\n", params.S, params.n, params.m);
        // update Snrms
        // compute_Snrms(params);

    }

    void learn() {
        mix_t params; 
        construct_mix_params(params);
        AdamSNet adam(params.n, params.m);

        for(int iter = 0; iter < 10; ++iter) {
            forward(params);
            backward(params);

            //dbgout2D("show_dS_sum:\n", params.dS_sum, params.n, params.m);

            compute_dS_sum(params);

            dbgout2D("show_dS_sum:\n", params.dS_sum, params.n, params.m);

            //adam.step(params.S, params.dS);
            update_S(params);

            // update Snrms
            compute_Snrms(params);

            //dbgout2D("after one round forward and backward, S:\n", params.S, params.n, params.m);
        }

    }
};

int main()
{
    // x XOR y =z
    std::vector<std::vector<bool>> assignments = {{true, true, false}, {false, false, false}, {true, false, true}, {false, true, true}};
    std::vector<int> input_dims = {1, 2};

    Learner L(std::move(input_dims), std::move(assignments), 4, 2);

    // with 2 aux variables, error become zero (although loss is not) after merely 3 iterations.
    L.learn();

    return 0;
}