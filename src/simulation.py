import numpy as np
from tqdm.notebook import trange, tqdm
import pickle
import pandas as pd
import os
   
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
            
        for ridx in iterator:
            row = included.iloc[ridx]
            model = pickle.load(open('./output/posteriors/model.p','rb'))
            samples = pickle.load(open(row['sample_loc'],'rb'))
            agg_df = pd.read_parquet(row[sim_row['data_location']])[row['start']:row['end']]
            follower_distrubtion = [np.array(item) for item in agg_df['follower_distribution'].values]
            
            outs = []
            
            for idx in range(sim_row['num_sims']):
                out = simulate(samples,
                         follower_distrubtion, nudge=sim_row['nudge'],
                         stop_at=sim_row['stop_at'], decay_value=sim_row['decay_value'], 
                         vcb_value = sim_row['vcb_value'], decay_start=sim_row['decay_start'],
                         p_decay=sim_row['p_decay'], p_remove=sim_row['p_remove'],
                         chain=np.random.choice(np.arange(samples['beta'].size)),
                         y=agg_df['total_tweets'].values)
                outs.append(out)
            out_dicts[row['event_name']] = np.vstack(outs)
        pickle.dump(out_dicts, open(save_loc, 'wb'))
        
        
def simulate(samples,follower_distribution,y,chain,nudge=1, 
             stop_at=np.inf, decay_value=1,vcb_value=1, decay_start=np.inf, p_decay=0,
             p_remove = 0,freq=5):
    
    start = np.min(np.where(y>0))
    alpha = samples['alpha'][chain]
    beta = samples['beta'][chain]
    phi = samples['phi'][chain]
    Lmbda = samples['lambda'][chain]
    decay = samples['decay'][chain]
    
    T = len(follower_distribution)
    output = np.zeros(T)
    
    if start > 0:
        output[0] = y[0:start] #Seed first timestep
    else: 
        output[0] = y[0]
    
    
    #Set up vcb and decay. We add 1 within the log as well in case
    #with banning there are no users in a given time-step (avoids divide by zero)
    virality = np.log(1+nudge*np.sum(1+follower_distribution[start]))
    vcb = 1
    will_decay = np.random.uniform(0,1) < p_decay
    will_remove = (np.random.uniform(0,1) < p_remove) and p_decay
    
    for t in range(start+1,T):
        
        if (t*freq > decay_start) and  will_decay: #Time Lagged Delay
            vcb = vcb_value 
        
        if (t*freq > stop_at) and will_remove: #Time lagged removal
            break

        eta_hat = alpha + beta * virality * vcb

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
        
    
