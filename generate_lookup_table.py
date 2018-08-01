# -*- coding: utf-8 -*-

import os
import time
import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from functions import sfh, generate_spectra, measure_spec, save_lookup

import fsps

from mangadap.util.instrument import spectrum_velocity_scale

from mangadap.drpfits import DRPFits
from mangadap.par.obsinput import ObsInputPar
from mangadap.proc.spatialbinning import RadialBinningPar, RadialBinning
from mangadap.proc.spectralstack import SpectralStackPar, SpectralStack

from mangadap.proc.templatelibrary import TemplateLibrary

from mangadap.proc.ppxffit import PPXFFit
from mangadap.proc.stellarcontinuummodel import StellarContinuumModelBitMask

from mangadap.proc.emissionlinemodel import EmissionLineModelBitMask
from mangadap.proc.elric import Elric
from mangadap.par.emissionlinedb import EmissionLineDB
from mangadap.par.spectralfeaturedb import SpectralFeatureDBDef
from astropy import constants as con
from astropy import units as un

from mangadap.proc.spectralindices import SpectralIndices
from mangadap.par.absorptionindexdb import AbsorptionIndexDB
from mangadap.par.bandheadindexdb import BandheadIndexDB

from scipy import interpolate
from astropy.cosmology import Planck15 

from itertools import product
from tqdm import tqdm, trange

from multiprocessing import Lock
global l
l = Lock()


np.set_printoptions(suppress=True, precision=4)

######################################################################################################
#
# August 1st 2018, RJS
#
# This script lets a user construct a look up table across defined star formation history parameters 
# in order to speed up the run time of the model optimisation step in SNITCH with emcee.
# See the functions.py file for the functions used to define a SFH, generate spectra and measure spectra. 
# These functions may be altered by a user for their own requirements. 
#
# If you have any questions please email rjsmethurst@gmail.com
#
######################################################################################################


if __name__ == "__main__":

    padova_zmet = np.array([0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.001, 0.0012, 0.0016, 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.012, 0.015, 0.019, 0.024, 0.03])
    padova_zmetsol = 0.019
    padova_solmet = pad_zmet/pad_zmetsol
    global zmets
    zmets = (np.linspace(0, 10, 11)*2 + 1).astype(int)

    global time_steps
    time_steps = Planck15.age(10**np.linspace(-0.824, -3.295, 15)).reshape(-1,1,1).value

    maxtq = np.log10(np.nanmax(time_steps)*1e9)

    global ages
    ages = np.flip(13.805 - 10**(np.linspace(7, 10.14, 100))/1e9, axis=0).reshape(-1,1,1)

    tqs = np.append(np.flip(time_steps.flatten()[0]- 10**(np.linspace(7, np.log10((time_steps.flatten()[0]-0.1)*1e9), 48))/1e9, axis=0), [time_steps.flatten()[0]-0.001, time_steps.flatten()[0]+0.1], axis=0)
    taus = 10**np.linspace(6, 9.778, 50)/1e9

    for n in range(len(time_steps)):

        # tqs needs to be a changeable array so that the logarithmic nature where the grid is finer before the observed time is conserved
        # no matter the time of observation. i.e. tq needs to change as t_obs changes. 
        tqs = np.append(np.flip(time_steps.flatten()[n]- 10**(np.linspace(7, np.log10((time_steps.flatten()[n]-0.1)*1e9), 48))/1e9, axis=0), [time_steps.flatten()[n]-0.001, time_steps.flatten()[n]+0.1], axis=0)
        taus = 10**np.linspace(6, 9.778, 50)/1e9


        tq = tqs.reshape(1,-1,1).repeat(len(taus), axis=2)
        tau = taus.reshape(1,1,-1).repeat(len(tqs), axis=1)

        sfr = sfh(tq, tau, age).reshape(age.shape[0], -1) 

        if os.path.isfile("spectrum_all_star_formation_rates_tobs_"+str(len(time_steps))+"_tq_"+str(len(tqs))+"_tau_"+str(len(taus))+"_Z_"+str(len(zmets))+"_newtqs.npy"):
            prev_fluxes = np.load("spectrum_all_star_formation_rates_tobs_"+str(len(time_steps))+"_tq_"+str(len(tqs))+"_tau_"+str(len(taus))+"_Z_"+str(len(zmets))+"_newtqs.npy", mmap_mode='r')
            print(len(prev_fluxes))
            if len(prev_fluxes)==(len(tqs)*len(taus)*len(time_steps.flatten())*len(zmets)):
                fluxes = prev_fluxes
                print("Loaded previously generated spectra array and it's the right length.")
            else:
                more_fluxes = np.array(list(map(generate_spectra, tqdm(list(sfr.T))))).reshape(-1, len(manga_wave))
                print("We're still generating more spectra, strap in for a wait...")
                fluxes = np.append(prev_fluxes, more_fluxes, axis=0)
                np.save("spectrum_all_star_formation_rates_tobs_"+str(len(time_steps))+"_tq_"+str(len(tqs))+"_tau_"+str(len(taus))+"_Z_"+str(len(zmets))+"_newtqs.npy", np.append(fluxes, more_fluxes, axis=0))

        else:
            print("You haven't generated spectra before for these SFH parameters, so this is going to take some time...")
            fluxes = np.array(list(map(generate_spectra, tqdm(list(sfr.T))))).reshape(-1, len(manga_wave))
            np.save("spectrum_all_star_formation_rates_tobs_"+str(len(time_steps))+"_tq_"+str(len(tqs))+"_tau_"+str(len(taus))+"_Z_"+str(len(zmets))+"_newtqs.npy", fluxes)
       


    st = time.time()

    chunk_length = 100
    idxs =np.arange(0, len(fluxes), chunk_length)
    for n in range(len(idxs)-1):
        print('Measuring spectra with indexes: ' idxs[n], ':', idxs[n+1])
        save_lookup(fluxes[idxs[n]:idxs[n+1]])

    print("Look up table generation took ", (time.time() - st)/60./60., " hours to complete.\n")




