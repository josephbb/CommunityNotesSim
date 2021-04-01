import numpy as np
import pandas as pd
import pickle

def get_adjusted_df(events, incidents,root='.'):
    temp = events[events['included']]
    idx = temp.groupby(['incident'])['observed_engagement'].transform(max) == temp['observed_engagement']
    max_events = temp[idx]
    max_events_incidents = pd.merge(max_events, incidents, left_on=['incident_name'], 
                                    right_on=['incident_name'])

    before = []
    after = []
    durings =[]
    for row in max_events_incidents.iterrows():
        raw_df = pd.read_parquet(root 
                 + '/data/timeseries/aggregated/' 
                 + row[1]['incident_name'] + '_raw.parquet')
        after.append(raw_df.iloc[row[1]['end']:]['total_tweets'].sum())
        before.append(raw_df.iloc[:row[1]['start']]['total_tweets'].sum())
        durings.append(raw_df.iloc[row[1]['start']:row[1]['end']]['total_tweets'].sum())

    max_events_incidents['before'] = before
    max_events_incidents['after'] =after
    max_events_incidents['during'] =durings
    return max_events_incidents

def print_engagment(max_events_incidents):
    rounder = lambda x: '{:0.2f}'.format(x*100)
    total = max_events_incidents['before'].sum() + \
        max_events_incidents['during'].sum() + \
       max_events_incidents['after'].sum()
    print('Total engagement')
    print(max_events_incidents['before'].sum() + \
        max_events_incidents['during'].sum()+ \
          max_events_incidents['after'].sum())

    print('Before Engagement Total, Percent')
    print(max_events_incidents['before'].sum())
    print(rounder(max_events_incidents['before'].sum()/total))

    print('During Engagement Total, Percent')
    print(max_events_incidents['during'].sum())
    print(rounder(max_events_incidents['during'].sum()/total ))

    print('After Engagement Total, Percent')
    print(max_events_incidents['after'].sum())
    print(rounder(np.sum(max_events_incidents['after'])/total ))
    
def sample_neg_binom(model, max_events_incidents):
    x_sim = np.logspace(np.log(np.min(max_events_incidents['observed_engagement'])-10),
                    np.log(np.max(max_events_incidents['observed_engagement'])+20),20,base=np.e)
    after = max_events_incidents['after']
    during = np.log(max_events_incidents['observed_engagement'])
    samples = model.sampling(data = {'after':after.values.astype('int'), 
                                           'during':during.values, 
                                            'N':during.size, 
                                             'x_sim':np.log(x_sim), 
                                             'S':x_sim.size})
    return samples

def get_plot_df(order, max_events_incidents,sim_df, samples, root='.'):
    colors  = []
    plot_dfs = []

    for val in order:
        sim_row = sim_df.iloc[val]
        colors.append(sim_row['color'])
        vals = pickle.load(open(root + '/output/simulations/' + sim_row['name'] +'.p', 'rb'))

        keys = [item for item in vals.keys() if item in max_events_incidents['event_name'].values]
        subsequents = []
        durings = []
        for key in keys:
            chains = np.random.choice(np.arange(4000), vals[key][:,-1].size)
            during = np.sum(vals[key],axis=1)
            observed_cascade = np.log(during+1)
            subsequent = np.exp(np.multiply( samples['beta'][chains], observed_cascade) + \
                                samples['sigma'][chains]/2)
            subsequents.append(subsequent)
            durings.append(during)
        during_posterior = np.sum(np.array(durings),axis=0)
        subsequent_posterior = np.sum(np.array(subsequents),axis=0)
        total = subsequent_posterior
        plot_dfs.append(pd.DataFrame({'name':np.repeat(sim_row['name'],total.size), 
         'total':total}))
        

    plot_df = pd.concat(plot_dfs)
    plot_df['total_adjusted'] = (plot_df['total'].values)/1e6

    plot_df = plot_df.replace({'modest_decay_only':'VCB', 
                      'modest_nudge_only': 'Nudge',
                      'modest_ban_only':'Ban', 
                      'baseline':'Base.',
                      'modest_0':'Mod.'})

    plot_df = plot_df.replace({'aggressive_decay_only':'VCB', 
                      'aggressive_nudge_only': 'Nudge',
                      'aggressive_ban_only':'Ban', 
                      'baseline':'Base.',
                      'aggressive_0':'Aggr.'})
    return plot_df, colors
    