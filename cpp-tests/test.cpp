

#include "satnet.h"
#include <vector>
#include <cstring>
#include <iostream>
#include <random>
#include <cstdalign>

std::default_random_engine generator;
std::normal_distribution<float> distribution(0.0,1.0);

struct MaxSAT {
  int nvar, mclauses, aux;
  std::vector<std::vector<float>> S; 
  std::vector<int> sol;

  
  MaxSAT(const std::vector<std::vector<int>>& clauses) {
    if (clauses.size() <= 0) {
      std::cerr << "cerror: lause size = " << clauses.size() << std::endl;
      return;
    }

    mclauses = clauses.size();
    nvar = clauses[0].size() + 1;

    for (int i = 0; i < mclauses; ++i) {
      std::vector<float> cl(nvar,-1.0); 
      for (int j = 1 ; j < nvar; ++j) {
        if (clauses[i][j-1] > 0) cl[j] = 1.0;
        if (clauses[i][j-1] == 0) cl[j] = 0.0;
      }
      S.push_back(std::move(cl));
    }
  }

  void show(){
    std::cout << "nvar = " << this->nvar << ", mclauses = " << this->mclauses << std::endl;

    for (int i =0; i < this->S.size(); ++i) {
      auto& v = this->S[i];
      for (int j = 0; j < v.size(); ++j) {
        std:: cout<<" "<< v[j];
      }
      std::cout << std::endl;
    }
  }

  void construct_mix_params(mix_t& params) {

    const int n = this->nvar;
    const int m = this->mclauses;
    const int k = 16; // embedding dimension size
    params.n = n;
    params.m = m;
    params.b = 1; // batch
    params.k = k; 

    params.is_input = new int32_t [n];
    memset(params.is_input, 0, sizeof(params.is_input));
    params.is_input[0] = 1;

    params.z = new float [n];
    params.z[0] = 1.0;
    for(int i = 1; i < n; ++i) {
      params.z[i] = 0.1;
    }
    params.dz = new float [n];
    memset(params.dz, 0, sizeof(params.dz));

    params.index = new int32_t [n];
    for (int i = 0; i < n; ++i) {
      params.index[i] = i;
    }


    params.niter = new int32_t (40); // allocate a single int with value 40

    params.S = new float [n * m];
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        const float sij = this->S[j][i];
        params.S[i * m + j] = sij;
      }
    }
    /*
    for (int j = 0; j < m; ++j) {
      float lits = 0.0;
      for (int i = 0; i < n; ++i) {
        const float sij = params.S[i * m + j];
        if (sij < -0.5 || sij > 0.5) lits += 1.0;
      }
      lits = 1.0/sqrtf32(4.0 * lits);
      for (int i = 0; i < n; ++i) {
        params.S[i * m + j] *= lits;
      }
    }*/
    params.Snrms = new float[n];
    for (int i = 0; i < n; ++i) {
      float r = 0.0;
      for (int j = 0; j < m; ++j) {
        const float sij = params.S[i * m + j];
        r += sij * sij;
      }
      params.Snrms[i] = r;
    }

    printf("Params.S: \n");
    for (int j = 0; j < m; ++j) {
      for (int i = 0; i < n; ++i) {
        printf("%.4f ", params.S[i*m + j]);
      }
      printf("\n");
    }
    printf("params.Snrms: \n");
    for (int i = 0; i < n; ++i) {
      printf("%.4f ", params.Snrms[i]);
    }
    printf("\n");


    params.dS = new float [n * m];
    memset(params.dS, 0, sizeof(params.dS));

    params.V = new float [n * k];
    params.U = new float [n * k];
    for (int i = 0; i < n * k; ++i) {
      params.V[i] = distribution(generator);
      params.U[i] = distribution(generator);
    }


    params.W = new float [k * m];


    params.Phi = new float [k * m];
    

    params.cache = new float [n];
    params.gnrm = new float [n];
    memset(params.cache, 0, sizeof(params.cache));
    // memset(params.gnrm, 0, sizeof(params.gnrm));
    for (int i = 0; i < n; ++i) {
      params.gnrm[i] =  distribution(generator);
    }
  }

  void prepare_W(mix_t& params){
    const int n = params.n;
    const int m = params.m;
    const int k = params.k;
    // W = V^T * S
    // should be perfomed AFTER V is normalized
    for (int i = 0; i < k; ++i) {
      for  (int j = 0; j < m; ++j) {
        float res = 0.0; 
        for (int d = 0; d < n; ++d) {
          res += params.V[d *k + i] * params.S[d * m + j];
        }
        int index = i * m  + j;
        params.W[index] = res;
      }
    }

    printf("params.V, shape: %d x %d\n", n, k);
    for (int i = 0; i< n; ++i) {
      for (int j = 0; j<k; ++j) {
        printf("%-7.4f ", params.V[i*k + j]);
      }
      printf("\n");
    }

    printf("params.W, shape: %d x %d\n", k, m);
    for (int i = 0; i< k; ++i) {
      for (int j = 0; j<m; ++j) {
        printf("%-7.4f ", params.W[i*m + j]);
      }
      printf("\n");
    }

  }

  void solve() {
    mix_t params; 
    construct_mix_params(params);
    mix_init_launcher_cpu(params, params.index);
    prepare_W(params);
    mix_forward_launcher_cpu(params, 10, 1e-10);


    for(int i = 0; i < params.n; ++i) {
      std::cout << " " << params.z[i]; 
    }
    std::cout << std::endl;
  }

};


int main() {
  // m has to be chosen properly, other 16-byte (4-float) SSE alignment will fail and then crash
  // std::vector<std::vector<int>> cnfs = {{1, 0, -1}, {0, 1, -1}, {1, 0, -1}, {0, 1, -1} };

  // parity rules

  std::vector<std::vector<int>> cnfs = {{-1, -1}, {1, 1}};
  MaxSAT ms(cnfs);

  ms.show();
  ms.solve();

  return 0;
}
