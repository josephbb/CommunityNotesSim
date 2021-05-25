import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def get_aggregated_follower_counts(group,removed):

    return group[~group['user_screen_name'].isin(removed)]['user_followers_count'].to_list()

def aggregate(raw_df, freq=5, removed=[], to_share=False):
    #Split raw_df at freq intervals
    raw_df['created_at'] =pd.DatetimeIndex(raw_df['created_at'])
    raw_df['total_tweets'] = np.ones(raw_df.shape[0])
    if to_share is True: 
        rounded = 10*np.round(raw_df['user_followers_count'].values  /10)
        raw_df['user_followers_count'] = rounded.astype('int')
    grouped = raw_df.groupby(pd.Grouper(key='created_at',freq=str(freq)+'Min'))

    #Apply banned user filter
    temp_funct = lambda x: get_aggregated_follower_counts(x, removed)
    fcs = grouped.apply(temp_funct)
    
    #Combine 
    aggregated = grouped.aggregate(np.sum)
    aggregated['follower_distribution'] = fcs
    return aggregated

def segment_ts(x,before_min=.05,freq=5, after_min=.05,pmin=.3):
    peaks = find_peaks(x, height=np.max(x)*pmin)#, distance=60/freq*distance_hours)
    
    start = 0
    peak_locs = np.array(peaks[0])
    peak_heights = peaks[1]['peak_heights']
    
    cuts = []
    upper = 0
    for idx in np.arange(peak_locs.size):            
        possible_lower = np.where(x[upper:peak_locs[idx]] 
                                  < before_min*peak_heights[idx])[0].tolist()
        possible_upper = np.where(x[peak_locs[idx]:-1]
                                  <after_min*peak_heights[idx])[0].tolist()
        
        
        if len(possible_lower) > 0 and len(possible_upper) > 0:
            lower=upper+np.max(possible_lower)
            upper = peak_locs[idx]+np.min(possible_upper)
            cuts.append([lower,upper])
            
    return cuts

def get_peaks(row,root='.', start=0, freq=5,pmin=.3,after_min=.05, before_min=.05):
    raw_df = pd.read_parquet(root + '/data/timeseries/aggregated/' 
                             + row['incident_name'] + '_raw.parquet')
    x = raw_df['total_tweets'].values
    peak_cuts=segment_ts(x,freq=5,pmin=.3,after_min=.05, 
                                before_min=.05)
    peak_cuts = [cut for cut in peak_cuts if cut[1]-cut[0] >= 12]
    return peak_cuts

def plot_event(row, freq=5, root='.'):
    raw_df = pd.read_parquet(root 
                         + '/data/timeseries/aggregated/' 
                         + row['incident_name'] + '_raw.parquet')
    y = raw_df['total_tweets'][row['start']:row['end']].values
    x = np.arange(y.size)*freq
    plt.figure()
    plt.title(row['event_name'])
    plt.plot(x,y,color='k')
    plt.ylabel('Tweets per '+str(freq) + 'Min')
    plt.xlabel('Time')
    plt.ylim(-freq/2, np.max(y)*1.1)
    plt.xlim(0, np.max(x))
    plt.savefig(root + '/output/figures/Events/' + row['event_name'] + '.png')
    plt.close()