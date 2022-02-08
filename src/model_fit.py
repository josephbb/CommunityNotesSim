import pystan
import numpy as np
import pandas as pd
import pickle
import os
os.environ['STAN_NUM_THREADS'] = "4"


def fit_model(row, model, keep=True,root='.'):
    if keep and os.path.isfile(root + '/output/posteriors/' + row['event_name'] + '_raw.p'):
        pass
    
    else:
        raw_df = pd.read_csv(root 
                     + '/data/timeseries/aggregated/' 
                     + row['incident_name'] + '_raw.csv')
        raw_df = raw_df.iloc[row['start']:row['end']]

        #We add one here because it avoids a divide by zero error in log. Should not impact inference
        stan_data = dict(y=raw_df['total_tweets'].values.astype('int')[1:],
                        N=raw_df.shape[0], 
                        x=np.log(raw_df['user_followers_count'].values.astype('int')+1))
        samples = model.sampling(data=stan_data,
                                 check_hmc_diagnostics=False,
                                 refresh = 0, 
                                 control={'adapt_delta':.99})
        pickle.dump(samples.extract(inc_warmup=False), 
            open(root + '/output/posteriors/' + row['event_name'] + '_extracted.p', 'wb'))
        pickle.dump(samples, 
            open(root + '/output/posteriors/' + row['event_name'] + '_raw.p', 'wb'))       
        return pystan.check_hmc_diagnostics(samples, pars={'alpha','beta','lambda','decay','phi'})
    
def check_fit(row,root='.',):
    output_dict = {}
    sample_loc = root+'/output/posteriors/' + row['event_name'] + '_raw.p'
    data_loc = root+'/data/timeseries/aggregated/' + row['incident_name'] + '_raw.csv'
    samples = pickle.load(open(sample_loc, 'rb'))
    dat = pd.read_csv(data_loc).iloc[row['start']:row['end']]
    y = dat['total_tweets'].values
    
    print('--------')
    print(row['event_name'])
    output_dict.update(pystan.check_hmc_diagnostics(samples, pars=['alpha','beta','decay', 
                                                 'lambda','phi']))
    

    observed_engagement = np.sum(y)
    output_dict['observed_engagement'] = observed_engagement

    output_dict['y_0'] = y[0]
    output_dict['observed_engagement'] = observed_engagement
    
    low, high = np.percentile(np.cumsum(samples['y_hat'],axis=1)[:,-1], q=[5.5,94.5])
    cumulative_fit = np.cumsum(y)[-1] > low and np.cumsum(y)[-1] < high


    output_dict['final_predicted' ] = cumulative_fit
    
    output_dict.update(row.to_dict())
    output_dict['sample_loc'] = sample_loc
    output_dict['data_loc'] = data_loc
    output_dict['lower_predicted']=low
    output_dict['upper_predicted']=high

    return output_dict