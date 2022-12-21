import numpy as np
from tqdm.notebook import trange, tqdm
import pickle
import pandas as pd
import os
import json
from ast import literal_eval
def sim_row(sim_row,included,keep=False, verbose=False):
    save_loc = './output/simulations/' + sim_row['name'] + '.p'
    if keep and os.path.isfile(save_loc):
        pass 
    else:       
        out_dicts = {}
        
        if verbose:  
            iterator = tqdm(range(included.shape[0]))
        else:
            iterator = range(included.shape[0])        
        
        is_largest = np.array(included['observed_engagement'] > np.percentile(included['observed_engagement'],(1-sim_row['p_nudge'])*100))
        
        for ridx in iterator:
            row = included.iloc[ridx]
            samples = pickle.load(open(row['sample_loc'],'rb'))
            agg_df = pd.read_csv(row[sim_row['data_location']])[row['start']:row['end']]
            follower_distribution = [np.array(literal_eval(item)) for item in agg_df['follower_distribution'].values]
            outs = []
            
            if sim_row['largest']: 
                if is_largest[ridx] == 1:

                    p_nudge = 1
                else:
                    p_nudge = 0
            else:
                p_nudge = sim_row['p_nudge']

            for idx in range(sim_row['num_sims']):
                out = simulate(samples,
                        follower_distribution, 
                        stop_at=sim_row['stop_at'], nudge_value=sim_row['nudge_value'], 
                        nudge_start=sim_row['nudge_start'],
                        p_nudge=p_nudge,
                        chain=[np.random.choice(np.arange(samples.posterior['beta'].shape[0])), 
                            np.random.choice(np.arange(samples.posterior['beta'].shape[1]))],
                        y=agg_df['total_tweets'].values)
                outs.append(out)
            out_dicts[row['event_name']] = np.vstack(outs)
        pickle.dump(out_dicts, open(save_loc, 'wb'))
        
        
def simulate(samples,follower_distribution,y,chain, stop_at=np.inf, p_nudge=1, nudge_value=1, nudge_start=np.inf,freq=5):
    start = np.min(np.where(y>0))
    alpha = np.array(samples.posterior['alpha'])[chain[0], chain[1]]
    beta = np.array(samples.posterior['beta'])[chain[0], chain[1]]
    phi = np.array(samples.posterior['phi'])[chain[0], chain[1]]
    Lmbda = np.array(samples.posterior['lambda'])[chain[0], chain[1]]
    decay = np.array(samples.posterior['decay'])[chain[0], chain[1]]
    nudge = 1

    T = len(follower_distribution)
    output = np.zeros(T)

    if start > 0:
        output[0] = y[0:start] #Seed first timestep
    else: 
        try:
            output[0] = y[0]
        except:
            output[0] = y.values[0]

    #Set up vcb and decay. We add 1 within the log as well in case
    #with banning there are no users in a given time-step (avoids divide by zero)
    virality = np.log(1+nudge*np.sum(1+follower_distribution[start]))
    will_nudge = np.random.uniform(0,1) < p_nudge

    for t in range(start+1,T):
        
        if (t*freq > nudge_start) and  will_nudge: #Time Lagged Delay
            nudge = nudge_value 

        eta_hat = alpha + beta * virality 

        #If no one is left to tweet in a timestep, rare
        if follower_distribution[t].size==0: 
            x_sim=0
        else:
            x_sim =np.log(1+nudge*np.sum(1+np.random.choice(follower_distribution[t],
                                size=np.round(output[t-1]).astype('int'))))
            
        virality = (virality)*(decay*np.power(np.e, -Lmbda*t))+x_sim
        mu = np.exp(eta_hat) 
        output[t] = np.random.negative_binomial(phi, phi/(mu+phi))

        #Prevent rare runaway from biasing results. 
        if output[t] > 2*np.max(y):
            output[t] = 2*np.max(y)
    return output