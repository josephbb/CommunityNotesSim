import pandas
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import numpy as np
import os
import glob
import string
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from src.utils import interp
from src.simulation import simulate
from ast import literal_eval
import json
import stan

def reshape(x):
    x = np.array(x)
    return np.reshape(x, (x.shape[0]*x.shape[1],x.shape[2]))

def plot_posterior(row, cumulative=True,freq=5,root='.', div=1000,color='k'):
    sample_loc = root+'/output/posteriors/' + row['event_name'] + '_raw.p'
    dat = root+'/data/timeseries/aggregated/' + row['incident_name'] + '_raw.csv'
    
    df = pd.read_csv(dat).iloc[row['start']:row['end']]
    samples = pickle.load(open(sample_loc,'rb'))
    
    x= np.arange(df.shape[0]-1)*freq
    y = df['total_tweets'].values[1:]
    
    mu = np.mean(reshape(samples.posterior['y_hat']),axis=0)
    ci = np.percentile(reshape(samples.posterior['y_hat']),q=[5.5,94.5],axis=0)
    
    if cumulative:
        y=np.cumsum(y)/div
        mu = np.median(np.cumsum(reshape(samples.posterior['y_hat']),axis=1),axis=0)/div

        ci = np.percentile(np.cumsum(reshape(samples.posterior['y_hat']),axis=1)
                           ,q=[5.5,94.5],axis=0)/div
        
    plt.plot(x,y ,color='k')
    plt.plot(x,mu,ls='--',color=color)
    plt.fill_between(x, ci[0,:], ci[1,:],alpha=.3,facecolor=color)
    
    
    plt.ylabel('Posts per %s min.' % str(freq))
    if cumulative:
        plt.ylabel('Cumulative posts \n(thousands)')
    plt.xlabel('Time (min.)')
    plt.xlim(0,mu.size*freq-5)

def plot_posterior_and_save(row,root='.', keep=True,freq=5,color='k'):
    file_output = root + '/output/figures/SI/posterior_predictive/' + row['event_name'] + '_pp.png'
    if row['included']==True:
        if keep and os.path.isfile(file_output):
            pass
        else:
            plt.subplot(211)
            samples = plot_posterior(row,cumulative=False, freq=5,color=color)
            plt.subplot(212)
            samples = plot_posterior(row,cumulative=True,freq=5,color=color)
            plt.suptitle(row['event_name'])
            plt.tight_layout()
            plt.savefig(file_output,dpi=300)
            plt.close()

def SI_Posterior(included, root='.'):
    for start in range(12):
        fig = plt.figure(figsize=(25.5,33))
        gs0 = gridspec.GridSpec(8,5, figure=fig)

        for idx in range(40):
            event = start*40 + idx

            if event >= included.shape[0]:
                break


            gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[idx])

            ax1 = fig.add_subplot(gs00[0, 0])
            plt.title(included.iloc[event]['event_name'])
            plot_posterior(included.iloc[event],cumulative=False)
            plt.xlabel('')
            plt.xticks([], [])
            ax2 = fig.add_subplot(gs00[1, 0])
            plot_posterior(included.iloc[event])

        plt.tight_layout()
        plt.savefig(root + '/output/figures/SI/SI_Fits_' + str(start*40)+'.png',dpi=300)
        plt.savefig(root + '/output/figures/SI/SI_Fits_' + str(start*40)+'.pdf',format='pdf')

        plt.close()
        
        
def add_sim_line(sim_row,scale=1e6,root='.'):
    vals = pickle.load(open(root + '/output/simulations/' + sim_row['name'] +'.p', 'rb'))
    keys = [key for key in vals.keys()]
    one_output = vals[keys[0]]

    interp_size = 20
    one_output = vals[keys[0]]
    out = np.zeros((one_output.shape[0], interp_size))

    for key in keys:
        one_output = vals[key]
        cumulative_scaled = np.apply_along_axis(interp,arr=np.cumsum(one_output,axis=1),axis=1)
        out+= cumulative_scaled

    mu = np.median(out,axis=0)/scale
    ci = np.percentile(out,axis=0, q=[5.5,94.5])/scale
    xvals = np.linspace(0,1,interp_size)
   
 
    color = json.loads(sim_row['color'])
    plt.plot(xvals, mu,color=color)
    plt.fill_between(xvals, ci[0], ci[1],alpha=.4,facecolor=color, label=sim_row['name'])
    
    return out

def plot_sims(order, sim_df, legend_title, legend_column):
    for idx in order:
        _ = add_sim_line(sim_df.iloc[idx])

    labels = sim_df.iloc[order][legend_column].values.astype('str').tolist()
    labels[0] = 'Base.'
    legend = plt.legend(labels=labels,title=legend_title,loc=2,prop={'size': 8})
    legend.get_title().set_fontsize('10')
    plt.ylim(0,11)
    plt.xlim(0,1)
    plt.xlabel('Time (normalized)')
    plt.ylabel('Cumulative posts')

 