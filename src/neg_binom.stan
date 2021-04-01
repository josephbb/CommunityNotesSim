data {
int<lower=1> N;
int<lower=1> S;
real during[N];
int after[N];
real x_sim[S];
}

parameters {
real beta;
real<lower=0> sigma;
}

model {
real eta[N]; 

beta ~ cauchy(0,1);
sigma ~ cauchy(0,1);

for (n in 1:N){
eta[n] = beta * during[n];
after[n] ~ lognormal(eta[n], sigma);
}

}

generated quantities {
real y_sim[S];
real eta_sim[S];

    for (s in 1:S) {
        eta_sim[s] = beta * x_sim[s];
        y_sim[s] = lognormal_rng(eta_sim[s], sigma);
    }


}