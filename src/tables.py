import pandas as pd
import numpy as np
import pickle 
from src.utils import interp
def get_sim_cumulative(sim_row, root='.'):
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
    return out

def get_table(sim_df, order, root='.'):
    baseline_out = get_sim_cumulative(sim_df.iloc[0])[:,-1]
    outrows = []
    for idx in order:
        out = get_sim_cumulative(sim_df.iloc[idx], root)[:,-1]
        mu = np.median(out)
        ci = np.percentile(out, q=[5.5,94.5])
        mu_percent = -(100-np.median(np.divide(out,baseline_out) * 100))
        ci_percent = -(100-np.percentile(np.divide(out,baseline_out) * 100,
                                                              q=[5.5,94.5]))
        outrows.append({'name':sim_df.iloc[idx]['name'],
         '$\mu (\% change)$':str(int(mu))+ " ("+'{0:.1f}'.format(mu_percent)+")", 
         '5.5\%':str(int(ci[0])) + " ("+'{0:.1f}'.format(ci_percent[0])+")",
         '94.5\%':str(int(ci[1]))+ " ("+'{0:.1f}'.format(ci_percent[1])+")"})
    table = pd.DataFrame(outrows)
    return table