import os
import numpy as np

def create_output_directories(location='.'):
    directories = ['output/','data/',
                   'data/timeseries/',
                   'data/timeseries/raw/',
                   'data/timeseries/processed/',
                   'output/',
                   'output/tables/',
                   'output/figures/',
                   'output/figures/Events/',
                   'output/figures/SI/posterior_predictive/',
                   'output/simulations/',
                   'output/posteriors/']
    
    for directory in directories:
        if not os.path.exists(location + '/' + directory):
            os.makedirs(location + '/' + directory)
            
def interp(data, newsize=20):
    size=data.size
    xloc = np.arange(size)
    new_xloc = np.linspace(0, size, newsize)
    new_data = np.interp(new_xloc, xloc, data)
    return(new_data)
