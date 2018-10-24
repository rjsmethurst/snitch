# snitch

<img align="left" width="350" src="/images/snitch_logo.001.jpeg">

Code to conduct Bayesian inference of star formation histories using measured emission and absorption features of a galaxy or IFU spectra. This code is described in Smethurst et al. (2018; in prep - see [here](https://trac.sdss.org/attachment/wiki/MANGA/Projects/agn_gradients/snitch.pdf) for SDSS members) and we encourage all users to read that paper before using this code. 


## Prerequisites

This code is written in Python 3. To run snitch you also need [emcee](http://dfm.io/emcee/current/user/install) and [python-fsps](http://dfm.io/python-fsps/current/installation/) installed. Please follow the instructions on the links provided to ensure that these are installed properly before continuing. 

Similarly, the MaNGA MPL-5 DAP python scripts need to be installed. These are currently only available for SDSS members at this [link](https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-5/dap/LocalInstall) (requires log in). 

## Installation 

Clone this repository, either through the GitHub desktop interface or with the following command:

```git clone https://github.com/rjsmethurst/snitch.git```

The python scripts are available in the snitch folder, so make sure to then:

``` cd snitch/snitch/```

Once the paper describing this code is accepted for publication we intend to publish this repo with a DOI so that both a paper and the code can be cited. 

## Basic Usage 

In it's default format, snitch needs 5 spectral parameters, their corresponding measurement errors, the observed redshift and an identification string to infer a SFH. These 5 parameters are the EW\[H⍺\], D<sub>n</sub>4000, Hβ, Hδ<sub>A</sub> and MgFe'. If you wish to use snitch with these parameters then from the command line run:

```python3 snitch.py ha e_ha d4000 e_d4000 hbeta e_hbeta hdeltaA e_hdeltaA mgfe e_mgfe redshift identification```

where e_xxx is the error on the spectal parameter xxx. For example:

```python3 snitch.py 15.1 1.3 1.45 0.3 2.3 0.1 1.7 0.3 2.6 0.3 0.015 'sdss-657'```

will print a similar result to the following:

``` Best fit Z value (3.s.f.) found by SNITCH for  sdss-657 input parameters are : [ 0.979, +0.377, -0.326 ]```

```Best fit t_q value (3.s.f.) found by SNITCH for  sdss-657 input parameters are : [ 12.420, +0.350, -0.478 ]```

```Best fit tau value (3.s.f.) found by SNITCH for  sdss-657 input parameters are : [ 0.651, +0.229, -0.162 ]```

and will produce plots showing the MCMC 'walker' steps with time and as a corner plot. 

---

## Look Up Table Generation

The basic snitch runs using a pre-generated look up table of the 5 spectral parameters listed above measured in model spectra generated using Charlie Conroy's FSPS models. This look up table is generated for 15 time of observation (i.e. redshift), 11 metallicities, 50 quenching rates and 50 quenching times (which change with every time of observation to ensure a finer grid for recent quenching times). 

We generate this look up table in the ranges: 11.85 < t<sub>obs</sub> \[Gyr\] < 13.79 (i.e 0.15 > z > 0.0005), 0.01 > Z \[Z<sub>☉</sub>\] > 1.27,  0.001 < τ \[Gyr\] < 6.0, 0.01 < t<sub>q</sub> \[Gyr\] < 13.89. 

If this look up table is not appropriate for your science case, perhaps your observations span a higher redshift range or you would prefer to work with another spectral parameter, then you will need to generate your own look up table. A user can do this by altering the `generate_lookup_table.py` and `functions.py` scripts. Note that generating a look up table can take a significant amount of time depending on how many sfh parameters it is generated over (for the 15x11x50x50 array generated for  the basic usage this took a week to generate the spectra and then measure the parameters). Outlined below are three scenarios in which a new look up table may be required.

### Changing the range of parameters for the look up table 

Lines 74-101 of `generate_lookup_table.py` define the SFH parameters for which spectra should be generated and measured. Change these arrays for your specific science case. 

### Changing which spectral parameters are returned

The `measure_spec` function in the `functions.py` script loads in the databases of absorption indices and emission lines that the MaNGA DAP functions use in the spectral fitting routine. These are defined in lines 177-191 of `functions.py` wherein the `extindxsnitch.par` and `elpsnitch.par` files are loaded by the MaNGA DAP. These files can be adapted to measure different spectral parameters but must follow the format for columns outlined at the top of each file. 

If you adapt snitch to return more than 5 parameters you will also have to adapt the functions in `snitch_mcmc.py` so that all the parameters in the look up table are used by [emcee](http://dfm.io/emcee/current/user/install) to infer the SFH. 

### Changing the spectral fitting procedure

The `measure_spec` function in the `functions.py` script details the spectral fitting routine using the MaNGA DAP functions. IF you have your own spectral fitting routine you can adapt lines 207-209. The `measure_spec` function returns a recorded array for the emission line and absorption index measurements which are then accessed to give the equivalent widths and indices in the `save_lookup` function in `functions.py`. Depending on what your spectral fitting routine returns you may also need to adapt what `measure_spec` returns and therefore what `save_lookup` does with the returned values. 

---

## Adapting for purpose

You may wish to adapt `snitch.py` so that the spectral measurements of many spectra are loaded from a table rather than input on the command line for each spectra. If this is the case edit the `snitch.py` file from line 105 onwards. Remove the argument parsers from the command line and place the call to snitch in line 136 in a loop over your table entries. 

---

If you have any other questions about running SNITCH or how to adapt it for your purpose beyond the basic usage please raise an issue in the repository (if you are not familar with GitHub [here's](https://help.github.com/articles/creating-an-issue/) some instructions on how to raise an issue) or contact me at rebecca dot smethurst AT physics dot ox dot ac dot uk.  
