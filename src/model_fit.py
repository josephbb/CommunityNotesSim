import stan
import numpy as np
import pandas as pd
import pickle
import os
import arviz as az
os.environ['STAN_NUM_THREADS'] = "4"


def fit_model(row, keep=True,root='.'):
    if keep and os.path.isfile(root + '/output/posteriors/' + row['event_name'] + '_raw.p'):
        pass
    
    else:
        with open('./src/timeseries.stan', 'r') as file:
            model_code = file.read() 
        raw_df = pd.read_csv(root 
                     + '/data/timeseries/aggregated/' 
                     + row['incident_name'] + '_raw.csv')
        raw_df = raw_df.iloc[row['start']:row['end']]

        #We add one here because it avoids a divide by zero error in log. Should not impact inference
        stan_data = dict(y=raw_df['total_tweets'].values.astype('int')[1:],
                        N=raw_df.shape[0], 
                        x=np.log(raw_df['user_followers_count'].values.astype('int')+1))
        model = stan.build(model_code, data=stan_data, random_seed=123)
        samples = az.from_pystan(model.sample())
        samples.to_json(root + '/output/posteriors/' + row['event_name'] + '_raw.json')
        with open(root + '/output/posteriors/' + row['event_name'] + '_raw.p', "wb") as f:
            pickle.dump(samples, f)
        temp = az.rhat(samples)
        return {'BFMI': np.all(az.bfmi(samples) > .3), \
            "rhat":np.all([temp.data_vars[item].values==True for item in temp.data_vars])}
    
    

def check_fit(row,root='.',):
    output_dict = {}
    output_dict.update(row.to_dict())

    sample_loc = root+'/output/posteriors/' + row['event_name'] + '_raw.p'
    print(sample_loc)
    data_loc = root+'/data/timeseries/aggregated/' + row['incident_name'] + '_raw.csv'
    samples = pickle.load(open(sample_loc, 'rb'))
    
    dat = pd.read_csv(data_loc).iloc[row['start']:row['end']]
    y = dat['total_tweets'].values
    print('--------')
    print(row['event_name'])
    temp = az.rhat(samples, var_names=['beta', 'decay','alpha','lambda','phi'])
    output_dict.update({'BFMI': np.all(az.bfmi(samples) > .3), \
            "rhat":np.all([temp.data_vars[item].values<1.1 for item in temp.data_vars])})
    

    observed_engagement = np.sum(y)
    output_dict['observed_engagement'] = observed_engagement
    output_dict['y_0'] = y[0]
    output_dict['observed_engagement'] = observed_engagement
    
 
    low, high = np.percentile(np.cumsum(samples.posterior['y_hat'],axis=2)[:,:,-1], q=[5.5,94.5])
    cumulative_fit = np.all([np.cumsum(y)[-1] > low, np.cumsum(y)[-1] < high])
    
    output_dict['final_predicted' ] = cumulative_fit
    output_dict['sample_loc'] = sample_loc
    output_dict['data_loc'] = data_loc
    output_dict['lower_predicted']=low
    output_dict['upper_predicted']=high
   
    
    return output_dict
