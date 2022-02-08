# CombinedPoliciesMisinfo
Combining interventions to reduce the spread of viral misinformation
<h1>System Requirements</h1>
<h2>Hardware</h2>
Our data and analysis were conducted on a linux server with sixty-four 2.6GHz CPUs and 256GB RAM. 
It has a 256GB primary disk partition on an RAID1 SSD and an 8TB secondary partition for home directories on a RAID6 storage array.
We believe this could be run on a laptop or smaller machine, but note that the run time may be substantial and
memory issues are likely. 
</h2>Software</h2> 
This work relies on Python3 and Pystan (2.19.1.1). .
This code *will fail* with a Pystan 3.X installation.
<h1>Replicating the analysis</h1>
<h2>Data access</h2>
Instructions for downloading the data are likewise provided on OSF
<h2>Installation</h2>
Detailed installation instructions are included on <a href=https://osf.io/2dcer/wiki/home/>OSF</a>. Installation time will vary depending upon
what you already have installed, typically no more than 30 minutes. 
<h2>Running the code</h2>
Running the code is the same as running any other Jupyter notebook. On our machine, it took approximately 8 hours using 
a fraction of the resources. It may take several hours/multiple days depending upon your system specifications
<h2>Replicating the results</h2>
The jupyter notebook reproduces all results exactly as they are presented in the paper. 
