/*
 * Modeling of cover class data
 * with spatial autocorrelation
 * using regularized incomplete beta function
 */

/*
 * sparse_iar_lpdf function by Max Joseph
 *
 * Exact sparse CAR models in Stan
 * https://mc-stan.org/users/documentation/case-studies/mbjoseph-CARStan.html
 */

functions {
  /**
  * Return the log probability of a proper intrinsic autoregressive (IAR) prior 
  * with a sparse representation for the adjacency matrix
  *
  * @param phi Vector containing the parameters with a IAR prior
  * @param tau Precision parameter for the IAR prior (real)
  * @param W_sparse Sparse representation of adjacency matrix (int array)
  * @param n Length of phi (int)
  * @param W_n Number of adjacent pairs (int)
  * @param D_sparse Number of neighbors for each location (vector)
  * @param lambda Eigenvalues of D^{-1/2}*W*D^{-1/2} (vector)
  *
  * @return Log probability density of IAR prior up to additive constant
  */
  real sparse_iar_lpdf(vector phi, real tau,
    int[,] W_sparse, vector D_sparse, vector lambda, int n, int W_n) {
      row_vector[n] phit_D; // phi' * D
      row_vector[n] phit_W; // phi' * W
      vector[n] ldet_terms;
    
      phit_D = (phi .* D_sparse)';
      phit_W = rep_row_vector(0, n);
      for (i in 1:W_n) {
        phit_W[W_sparse[i, 1]] = phit_W[W_sparse[i, 1]] + phi[W_sparse[i, 2]];
        phit_W[W_sparse[i, 2]] = phit_W[W_sparse[i, 2]] + phi[W_sparse[i, 1]];
      }
    
      return 0.5 * ((n-1) * log(tau)
                    - tau * (phit_D * phi - (phit_W * phi)));
  }

 /*
  * Return the log probability that the cover class is oberved under the
  * given paramters a and b.
  *
  * @param Y  Observed class
  * @param CP Cut points
  * @param a  Parameter of beta distribution
  * @param b  Parameter of beta distribution
  */
  real coverclass_lpmf(int Y, vector CP, real a, real b) {
    int n_cls;
    real gamma;
    n_cls = num_elements(CP) + 1;
    if (Y <= 1) {  // 0 or 1
      gamma =  inc_beta(a, b, CP[1]);
    } else if(Y >= 2 && Y < n_cls) {
      gamma = inc_beta(a, b, CP[Y])
              - inc_beta(a, b, CP[Y - 1]);
    } else {
      gamma = 1 - inc_beta(a, b, CP[n_cls - 1]);
    }
    return bernoulli_lpmf(1 | gamma);
  }

  int coverclass_rng(vector CP, int n_cls, real a, real b) {
    vector[n_cls] pr;
    int y;
    
    pr[1] = inc_beta(a, b, CP[1]);
    for (i in 2:(n_cls - 1))
      pr[i] = inc_beta(a, b, CP[i]) - inc_beta(a, b, CP[i - 1]);
    pr[n_cls] = 1 - inc_beta(a, b, CP[n_cls - 1]);
    y = categorical_rng(pr);
    return y;
  }
}

data {
  int<lower = 1> N_cls;                       // Number of classes
  int<lower = 1> N;                           // Number of observations
  int<lower = 0, upper = N_cls> Y[N];         // Observed cover class
  vector<lower = 0, upper = 1>[N_cls - 1] CP; // Cut points
  matrix<lower = 0, upper = 1>[N, N] W;       // Adjacency matrix
  int W_n;                                    // Number of adjacent region pairs
}

transformed data {
  int W_sparse[W_n, 2];   // adjacency pairs
  vector[N] D_sparse;     // diagonal of D (number of neigbors for each site)
  vector[N] lambda;       // eigenvalues of invsqrtD * W * invsqrtD
  
  { // generate sparse representation for W
  int counter;
  counter = 1;
  // loop over upper triangular part of W to identify neighbor pairs
    for (i in 1:(N - 1)) {
      for (j in (i + 1):N) {
        if (W[i, j] == 1) {
          W_sparse[counter, 1] = i;
          W_sparse[counter, 2] = j;
          counter = counter + 1;
        }
      }
    }
  }
  for (i in 1:N) D_sparse[i] = sum(W[i]);
  {
    vector[N] invsqrtD;  
    for (i in 1:N) {
      invsqrtD[i] = 1 / sqrt(D_sparse[i]);
    }
    lambda = eigenvalues_sym(quad_form(W, diag_matrix(invsqrtD)));
  }
}

parameters {
  real<lower = 0, upper = 1> delta;           // Uncertainty
  vector[N] phi;                              // Spatial random effect
  real<lower = 0> tau;                        // Parameter
}

transformed parameters {
  vector[N] mu = inv_logit(phi);             // Mean proportion of cover
}

model {
  // Spatial random effects
  phi ~ sparse_iar(tau, W_sparse, D_sparse, lambda, N, W_n);

  // Observation model
  for (n in 1:N) {
    real a = mu[n] / delta - mu[n];
    real b = (1 - mu[n]) * (1 - delta) / delta;

    Y[n] ~ coverclass(CP, a, b);
  }

  // Priors
  tau ~ gamma(2, 2);
}

generated quantities {
  int yrep[N];
  
  for (n in 1:N) {
    real a = mu[n] / delta - mu[n];
    real b = (1 - mu[n]) * (1 - delta) / delta;
    
    yrep[n] = coverclass_rng(CP, N_cls, a, b);
  }
}
