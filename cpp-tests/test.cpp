

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
    const int k = 32; // embedding dimension size
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
        params.S[i * m + j] = this->S[j][i];
      }
    }
    params.dS = new float [n * m];
    memset(params.dS, 0, sizeof(params.dS));

    params.V = new float [n * k];
    params.U = new float [n * k];
    for (int i = 0; i < n * k; ++i) {
      params.V[i] = distribution(generator);
      params.U[i] = distribution(generator);
    }


    params.W = new float [k * m];

    // W = V^T * S
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

    params.Phi = new float [k * m];
    

    params.cache = new float [n];
    params.gnrm = new float [n];
    params.Snrms = new float[n];
    memset(params.cache, 0, sizeof(params.cache));
    // memset(params.gnrm, 0, sizeof(params.gnrm));
    // memset(params.Snrms, 0, sizeof(params.Snrms));
    for (int i = 0; i < n; ++i) {
      params.gnrm[i] =  distribution(generator);
      params.Snrms[i] =  distribution(generator);
    }
  }

  void solve() {
    mix_t params; 
    construct_mix_params(params);
    mix_init_launcher_cpu(params, params.index);
    mix_forward_launcher_cpu(params, 40, 1e-10);


    for(int i = 1; i < params.n; ++i) {
      std::cout << " " << params.z[i]; 
    }
    std::cout << std::endl;
  }

};


int main() {
  // m has to be chosen properly, other 16-byte (4-float) SSE alignment will fail and then crash
  std::vector<std::vector<int>> cnfs = {{1, 0, -1}, {0, 1, -1}, {1, 0, -1}, {0, 1, -1} };

  MaxSAT ms(cnfs);

  ms.show();
  ms.solve();

  return 0;
}
