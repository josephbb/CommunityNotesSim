# [Limiting Factors in the Effectiveness of Crowd-Sourced Labeling for Combating Misinformation}](https://github.com/josephbb/CommunityNotesSim)
Joseph B. Bak-Coleman(1), Zeynep Tufekci(1),
1. Columbia University School of Journalism 



## Repository information
This repository provides all code and data necessary to generates results, tables, and figures found in the article ["Limiting Factors in the Effectiveness of Crowd-Sourced Labeling for Combating Misinformation"](https://github.com/josephbb/CommunityNotesSim) (J. Bak-Coleman et al. Forthcoming).

## Article abstract
Crowd-sourced labeling has been proposed as a solution for reducing the spread of misinformation on social media. Although there is evidence that labeling misinformation can reduce engagement at an individual level, we lack insight into whether crowd-sourced labels can meaningfully reduce misinformation at scale. To estimate the efficacy of crowd-sourced labeling, we adapted a computational model of viral misinformation spread parameterized by a corpus of 12.9 million tweets from the 2020 US presidential election. Our simulations reveal that individual-level reductions in engagement from crowd-sourced labels did not directly translate into commensurate reductions in the spread of misinformation. This is due to a number of factors, including the time required to surface a helpful note, the proportion of tweets labeled, and the extent to which labeled tweets target topics highly-engaging content. However, even if crowd-sourced labels are prompt and well-targeted, crowd-sourced labels are unlikely to be as effective as combining more traditional forms of content moderation. Overall, crowd-sourced labeling-based may be better used as a supplement to traditional content moderation rather than a standalone replacement. 

## License and citation information
If you plan on using this code for any purpose, please see the [license](LICENSE.txt) and please cite our work as below:

Citation and BiBTeX record to come.

## Directories
- ``Analysis.ipynb`` : Primary analysis file as an ipython notebook.
- ``analysis.py``: Python script (``.py``) export of polarization-analysis.ipynb.
- ``src``: The Bayesian Models (``*.stan``), code used to clean the raw data, code for generating figures, and utilities used in the primary analysis.  
    - ``adjustment.py`` functions to generate figures
    - ``timeseries.stan`` Stan model code
    - ``figures.py`` Misc lotting functions
    - ``segmentation.py`` Code for segmenting timeseries into events
    - ``simulation.py`` Simulation code
    - ``tables.py`` Code for generating tables
    - ``utils.py`` Miscellaneous utilities.
- ``data``: data files in comma-separated values (``.csv``) formats, downloaded from [zenodo](https://zenodo.org/record/6480218)
    - ``dat/timeseries``: timeseries data 
    - ``dat/incidents.csv``: List of all incidents
- ``out``: output files
    - ``out/posteriors``: Markov Chain Monte Carlo (MCMC) output, as pickled python objects (``.p``)
    - ``out/figures``: Figures generated from results
    - ``out/simulations``: Simulation results

## Reproducing analysis

You can reproduce the analysis, including all figures and tables by following the guide below. Please note that minor, non-qualitative differences may exist due to difference in pseudorandom number generation.

### Getting the code
First download this repository. Either download directly or open a command line and type:

    git clone https://github.com/josephbb/polarized-collective-wisdom

## Dependency installation guide
You will an [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or python installation and command-line interface. The simplest way to install the requirements is to navigate to the directory and type ``pip install -r requirements.txt``. You may, however, wish to install these in a [virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to avoid conflicts with your currently installed python packages. Note that installing these packages, particularly Stan and Pystan can take time and require compilation on your local machine.

### Getting the data
Download the data, which was published previously, from [Zenodo](https://zenodo.org/record/6480218). Extract it into a folder named ``data`` in the same directory as ``analysis.ipynb``
### Running the analysis

The simplest approach is to navigate to the directory and simply type:

    jupyter nbconvert --execute ./Analysis.ipynb --ExecutePreprocessor.timeout -1
    
This will generate a rendered output of the notebook(``.HTML``) that you can open in your browswer, along with all figures and tables on your local machine. Please note that this code can take a long time (perhaps hours) to run, necessitating  timeout being set to -1 in the command above.  . You may prefer simply to open and review the notebook using

    jupyter notebook


## Project structure when complete

Once the full analysis has been run, manuscript figures can be found in ``out/figures/MS``, and posteriors in ``out/posteriors``, figures depicting model fit in ``out/figures/Events``, and compilations of these figures in ``out/figures/SI`` 

#System Specifications

Beyond what is in requirements.txt, this analysis was run on a machine with the following configuration.

- CPU: Apple M1 Pro
- Memory: 16 GiB
- OS: MacOS Montery 12.5.1
- Python: 3.9.0
- Anaconda: 22.9.0
- Pystan 3.5.0 (Pystan 2 will not work)
- clang: Apple clang version 13.1.6
