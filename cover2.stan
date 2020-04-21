/*
 * Modeling of cover class data
 * using regularized incomplete beta function
 *
 * Model with covariate
 */

functions {
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
  int<lower = 1> R;                           // Number of replications
  int<lower = 0, upper = N_cls> Y[N, R];      // Observed cover class
  vector<lower = 0, upper = 1>[N_cls - 1] CP; // Cut points
  vector[N] X;                                // explanatory variable
}

parameters {
  real<lower = 0, upper = 1> delta;           // Intra-quad corr.
                                              //  or uncertainty
  vector[2] beta;                             // Intercept and coeff.
  vector[N] err;                              // Error in system (reparam)
  real<lower = 0> sigma;                      // SD of error
}

transformed parameters {
  vector<lower = 0, upper = 1>[N] mu;          // Mean proportion of cover

  // System
  mu = inv_logit(beta[1] + beta[2] * X + sigma * err);
}

model {
  // Observation
  for (n in 1:N) {
    real a = mu[n] / delta - mu[n];
    real b = (1 - mu[n]) * (1 - delta) / delta;

    for (r in 1:R)
      Y[n, r] ~ coverclass(CP, a, b);
  }
  // System
  err ~ std_normal();
  sigma ~ normal(0, 5);
}

generated quantities {
  int yrep[N];
  
  for (n in 1:N) {
    real mu_new;
    real a;
    real b;
    
    mu_new = inv_logit(beta[1] + beta[2] * X[n]
                      + normal_rng(0, sigma));
    a = mu_new / delta - mu_new;
    b = (1 - mu_new) * (1 - delta) / delta;
    
    yrep[n] = coverclass_rng(CP, N_cls, a, b);
  }
}
