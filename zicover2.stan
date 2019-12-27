/*
 * Modeling of cover class data
 * with zero-inflation
 * using regularized incomplete beta function
 */

functions {
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
    if (Y <= 1) {
      gamma =  inc_beta(a, b, CP[1]);
    } else if(Y >= 2 && Y < n_cls) {
      gamma = inc_beta(a, b, CP[Y])
              - inc_beta(a, b, CP[Y - 1]);
    } else {
      gamma = 1 - inc_beta(a, b, CP[n_cls - 1]);
    }
    return bernoulli_lpmf(1 | gamma);
  }

 /*
  * Return cover class randomly given cutpoints and paramters a and b.
  *
  * @param CP     Cut points
  * @param n_cls  Number of classes
  * @param a      Parameter of beta distribution
  * @param b      Parameter of beta distribution
  */
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
}

parameters {
  real<lower = 0, upper = 1> p;               // proportion of cover
  real<lower = 0, upper = 1> delta;           // intra-quad corr. or uncertainty
  real<lower = 0, upper = 1> omega;           // Pr(Y > 0)
  real<lower = 0, upper = 1> psi;             // Detection prob.
}

model {
  // Observation model
  {
    real a = p / delta - p;
    real b = (1 - p) * (1 - delta) / delta;
  
    for (n in 1:N) {
      if (Y[n] == 0) {
        real lp[2];
        
        lp[1] = bernoulli_lpmf(0 | omega);
        lp[2] = bernoulli_lpmf(1 | omega)
                + coverclass_lpmf(1 | CP, a, b)
                + bernoulli_lpmf(0 | psi);
        target += log_sum_exp(lp);
      } else if (Y[n] == 1) {
        target += bernoulli_lpmf(1 | omega)
                 + coverclass_lpmf(1 | CP, a, b)
                 + bernoulli_lpmf(1 | psi);
      } else {
        target += bernoulli_lpmf(1 | omega)
                 + coverclass_lpmf(Y[n] | CP, a, b);
      }
    }
  }
}

generated quantities {
  real phi = omega * p;
  int yrep[N];
  {
    real a = p / delta - p;
    real b = (1 - p) * (1 - delta) / delta;
    
    for (n in 1:N)
      if (bernoulli_rng(omega))
        yrep[n] = coverclass_rng(CP, N_cls, a, b);
      else
        yrep[n] = 0;
  }
}
