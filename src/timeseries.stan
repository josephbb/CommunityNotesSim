data {
int N;
int y[N];
real x[N];
}


parameters {
  real beta;
  real<lower=0, upper=1> decay;
  real alpha;
  real<lower=0> phi;
  real<lower=0> lambda;
}


model { 
real exposure[N];
real eta[N];
beta ~ normal(0,3);
alpha ~ normal(-3,3);
phi ~ exponential(1);
decay ~ beta(1,1);
lambda ~ exponential(1);

exposure[1] = x[1];

    for (n in 2:N){
    
        eta[n] = alpha + beta *exposure[n-1]; 
        exposure[n] = exposure[n-1]*decay*pow(e(),-lambda *n)  + x[n];
        y[n] ~ neg_binomial_2_log(eta[n], phi);
    }
}

generated quantities {
int y_hat[N];
real eta_hat[N];
real exposure_hat[N];
exposure_hat[1] = x[1];
y_hat[1] = y[1];
  for(n in 2:N){
        eta_hat[n] = alpha + beta*exposure_hat[n-1]; 
        exposure_hat[n] = exposure_hat[n-1]*decay*pow(e(),-lambda *n)+ x[n];
        if (eta_hat[n] > 14){
            y_hat[n] = -1;
        }
        else {
            y_hat[n] = neg_binomial_2_log_rng(eta_hat[n], phi);

        }
    }

}
