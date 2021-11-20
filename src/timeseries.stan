data {
int N;
int y[N-1];
real x[N];
}


parameters {
  real beta;
  real<lower=0, upper=1> decay;
  real alpha;
  real<lower=0> inv_phi;
  real<lower=0> lambda;
}

transformed parameters {
  real<lower=0> phi;
  phi = inv(inv_phi);
}

model { 
real exposure[N];
real eta[N];
beta ~ normal(0,3);
alpha ~ normal(-3,3);
inv_phi ~ normal(0,1);
decay ~ beta(1,1);
lambda ~ exponential(1);
exposure[1] = x[1];

    for (n in 2:N){
    
        eta[n] = alpha + beta *exposure[n-1]; 
        exposure[n] = exposure[n-1]*decay*pow(e(),-lambda *n)  + x[n];
        y[n-1] ~ neg_binomial_2_log(eta[n], phi);
    }
}

generated quantities {
int y_hat[N-1];
real eta_hat[N];
real exposure_hat[N];
real log_lik[N-1];
exposure_hat[1] = x[1];

  for(n in 2:N){
        eta_hat[n] = alpha + beta*exposure_hat[n-1]; 
        exposure_hat[n] = exposure_hat[n-1]*decay*pow(e(),-lambda *n)  + x[n];
        log_lik[n-1] = neg_binomial_2_log_lpmf(y[n-1] | eta_hat[n], phi);

        if (eta_hat[n] > 14){
            y_hat[n-1] = -1;
        }
        else {
            y_hat[n-1] = neg_binomial_2_log_rng(eta_hat[n], phi);

        }
    }


}
