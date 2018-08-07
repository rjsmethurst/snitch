# snitch

![snitchlogo](/images/snitch_logo.001.jpeg)

Code to conduct Bayesian inference of star formation histories using measured emission and absorption features of a galaxy or IFU spectra. This code is written in Python 3. 

## Prerequisites

To run snitch you also need [emcee](http://dfm.io/emcee/current/user/install) and [python-fsps](http://dfm.io/python-fsps/current/installation/) installed. Please follow the instructions on the links provided to ensure that these are installed properly before continuing. 

Similarly, the MaNGA MPL-5 DAP python scripts need to be installed. These are currently only available for SDSS members at this [link](https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-5/dap/LocalInstall) (requires log in). 

## Installation 

Clone this repository, either through the GitHub desktop interface or with the following command:

```git clone https://github.com/rjsmethurst/snitch.git```

The python scripts are available in the snitch folder, so make sure to then:

``` cd snitch/snitch/```

## Basic Usage 

In it's default format, snitch needs 5 spectral parameters, their corresponding measurement errors, the observed redshift and an identification string to infer a SFH. These 5 parameters are the EW\[H$\alpha$\], Dn4000, H$\beta$, H$\delta_A$ and MgFe'. If you wish to use snitch with these parameters then from the command line run:

```python3 snitch.py ha e_ha d4000 e_d4000 hbeta e_hbeta hdeltaA e_hdeltaA mgfe e_mgfe redshift identification```

for example:

```python3 snitch.py 15.6 1.3 2.3 0.3 2 0.1 3.2 0.3 1.7 0.3 0.015 'sdss-657'```

will print the following result:

```   Best fit [Z, tq, tau] values found by SNITCH for input parameters are : [Z_mcmc, tq_mcmc, tau_mcmc]```

and will produce plots showing the MCMC 'walker' steps with time and as a corner plot. 

## Look Up Table Generation



## Adapting for purpose



