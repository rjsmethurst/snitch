import os
import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from itertools import product
from scipy import interpolate
import fsps

import mangadap

from mangadap.proc.templatelibrary import TemplateLibrary
from mangadap.proc.ppxffit import PPXFFit
from mangadap.proc.stellarcontinuummodel import StellarContinuumModelBitMask

from mangadap.proc.emissionlinemodel import EmissionLineModelBitMask
from mangadap.proc.elric import Elric
from mangadap.par.emissionlinedb import EmissionLineDB
from mangadap.par.spectralfeaturedb import SpectralFeatureDBDef

from mangadap.proc.spectralindices import SpectralIndices
from mangadap.par.absorptionindexdb import AbsorptionIndexDB
from mangadap.par.bandheadindexdb import BandheadIndexDB

from astropy.cosmology import Planck15 
from astropy import constants as con
from astropy import units as un
c = con.c.to(un.km/un.s).value

from multiprocessing import Lock
global l
l = Lock()

######################################################################################################
#
# August 1st 2018, RJS
#
# This file contains all the functions needed to generate synthetic spectra for a defined SFH using 
# FSPS, measure these spectra using the MaNGA DAP and save it for later use as a look up table. 
# A user may wish to adapt these functions, either to define their own star formation histories, or 
# if they wish to use another method of spectral fitting to give spectral parameters. 
#
# Documentation is provided but if you have any further questions please email rjsmethurst@gmail.com
#
######################################################################################################

manga_wave = np.load("manga_wavelengths_AA.npy")

def expsfh(tq, tau, time):
    """ This function when given an array of [tq, tau] values will calcualte the SFR at all times. 
    First calculate the sSFR at all times as defined by Peng et al. (2010) - then the SFR at the specified time of 
    quenching, tq and set the SFR at this value  at all times before tq. Beyond this time the SFR is an exponentially 
    declining function with timescale tau. 
        
        INPUT:
        :tau:
        The exponential timescale decay rate of the star formation history in Gyr. Allowed range from the rest of the functions is 0 < tau [Gyr] < 5. Shape (1, M, N). 
        
        :tq: 
        The time at which the onset of quenching begins in Gyr. Allowed ranges from the beginning to the end of known cosmic time. Shape (1, M, N). 
        
        e.g:
        
        tqs = np.linspace(0, 14, 10)
        taus = np.linspace(0, 4, 5)
        
        tq = tqs.reshape(1,-1,1).repeat(len(taus), axis=2)
        tau = taus.reshape(1,1,-1).repeat(len(tqs), axis=1)
        
        :time:
        An array of time values at which the SFR is calcualted at each step. Shape (T, 1, 1)
        
        RETURNS:
        :sfr:
        Array of the same dimensions of time containing the sfr at each timestep. Shape (T, M, N). 

    A user can define their own SFH function here. Just make sure that the array returned is of the shape (T, M, N) as outlined above.

        """

    ssfr = 2.5*(((10**10.27)/1E10)**(-0.1))*(time/3.5)**(-2.2)
    c = np.apply_along_axis(lambda a: a.searchsorted(3.0), axis = 0, arr = time)    
    ssfr[:c.flatten()[0]] = np.interp(3.0, time.flatten(), ssfr.flatten())
    c_sfr = np.interp(tq, time.flatten(), ssfr.flatten())*(1E10)/(1E9)
    ### definition is for 10^10 M_solar galaxies and per gyr - convert to M_solar/year ###
    sfr = np.ones_like(time)*c_sfr
    mask = time <= tq
    sfrs = np.ma.masked_array(sfr, mask=mask)
    times = np.ma.masked_array(time-tq, mask=mask)
    sfh = sfrs*np.exp(-times/tau)
    return sfh.data


def generate_spectra(params):

    """ 
    This function allows the user to generate many spectra for a defined SFR at given ages, at user defined time steps and metallicities. 

    INPUT:
        :sfr:
        The SFRs defined at each age for any number of sfhs, H. Shape (H, SFR)
       
    GLOBAL PARAMETERS: 
        :ages: 
        The times at which the sfr is defined for any number of sfhs, H. Shape (H, SFR). 
        
        :time_steps:
        An array of time values for which a spectrum with the given sfr is returned. Shape (T,)

        :zmets:
        An array of model metallicity values (range 1-22, see FSPS details) for which a spectrum with the given sfr is returned. Shape (Z,)
        
    RETURNS:
        :fsps_flux:
        Fluxes at each manga wavelength, M for the number of sfhs input at each time_step and zmet combination. Shape (H*T*Z, M). 

    """
    time_step = [params[0]]
    zmets = params[1]
    ages = params[2]
    sfr = params[3]

    sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=0, zmet=18, sfh=3, dust_type=2, dust2=0.2, 
        add_neb_emission=True, add_neb_continuum=False, imf_type=1, add_dust_emission=True, min_wave_smooth=3600, max_wave_smooth=7500, smooth_velocity=True, sigma_smooth=77.)

    sp.set_tabular_sfh(age = ages, sfr=sfr)

    def time_spec(params):
        sps = sp.get_spectrum(tage=params[0], zmet=params[1])[1]
        return sps

    fsps_wave = sp.get_spectrum()[0]
    fsps_spec = np.array(list(map(time_spec, list(product(time_step, zmets)))))

    manga_wave = np.load("manga_wavelengths_AA.npy")
    manga_hz = c*(un.km/un.s).to(un.AA/un.s)/manga_wave

    f = interpolate.interp1d(fsps_wave, fsps_spec)
    fluxes = f(manga_wave)*(manga_hz/manga_wave) # Fluxes returned by FSPS are in units L_sol/Hz - MaNGA DAP needs fluxes in per AA flux units

    return fluxes 


def measure_spec(fluxes, errors=None, ivar=None, sres=None):
    """ This function takes an array of spectra of shape (# of spectra, # of wavelengths) and returns the pre-defined emission line and 
    absorption features in each spectra in two recorded arrays of shape (# of spectra,). The recorded arrays can be accessed to give the
    quantities returned by the MaNGA DAP, e.g. ["EW"], ["EWERR"], ["INDX"]. 

    The user can change the emission and absorption quantities returned by specifying different emission line and absorption feature 
    databases.

    INPUTS
    :fluxes:
    Fluxes at each manga wavelength, M for any number of spectra, X. Shape (X, M).

    OUTPUTS
    :em_model_eml_par: 
    Recorded array containing the emisison line parameter measurements from the defined user database. Shape (X, ).

    :indx_measurement:
    Recorded array containing the absorption feature parameter measurements from the defined user database. Shape (X, ).

    """
    if ivar is None:
        ivar = 1/(0.1*fluxes)**2
    if errors is None:
        errors = 0.1*fluxes
    if sres is None:
        sres = np.ones_like(fluxes)

    tpl = TemplateLibrary("MILESHC",
                            match_to_drp_resolution=False,
                            velscale_ratio=1,    
                            spectral_step=1e-4,
                            log=True,
                            directory_path=".",
                            processed_file="mileshc.fits",
                            clobber=True)

    # Instantiate the object that does the fitting
    contbm = StellarContinuumModelBitMask()
    ppxf = PPXFFit(contbm)

    # Define the absorption and bandhead feature databases
    define_abs_db = SpectralFeatureDBDef(key='USERABS',
                              file_path='extindxsnitch.par')
    abs_db = AbsorptionIndexDB(u"USERABS", indxdb_list=define_abs_db)

    band_db = BandheadIndexDB(u"BHBASIC")
    global indx_names
    indx_names = np.hstack([abs_db.data["name"], band_db.data["name"]])

    # Define the emission line feature database
    specm = EmissionLineModelBitMask()
    elric = Elric(specm)
    global emlines
    define_em_db = SpectralFeatureDBDef(key='USEREM',
                              file_path='elpsnitch.par')
    emlines  = EmissionLineDB(u"USEREM", emldb_list=define_em_db)
    
    # Check to see if a single spectra has been input. If the single spectra is shape (# of wavelengths,) then reshape 
    # to give (1, #number of wavelengths)
    if fluxes.shape[0] == len(manga_wave):
        fluxes = fluxes.reshape(1,-1)
    else:
        pass
    nspec = fluxes.shape[0]


    # Provide the guess redshift and guess velocity dispersion
    guess_redshift = np.full(nspec, 0.0001, dtype=float)
    guess_dispersion = np.full(nspec, 77.0, dtype=float)

    # Perform the fits to the continuum, emission lines and absorption features
    model_wave, model_flux, model_mask, model_par  = ppxf.fit(tpl["WAVE"].data.copy(), tpl["FLUX"].data.copy(), manga_wave, fluxes, errors, guess_redshift, guess_dispersion, iteration_mode="none", velscale_ratio=1, degree=8, mdegree=-1, moments=2, quiet=True)
    em_model_wave, em_model_flux, em_model_base, em_model_mask, em_model_fit_par, em_model_eml_par = elric.fit(manga_wave, fluxes, emission_lines=emlines, ivar=ivar, sres=sres, continuum=model_flux, guess_redshift = model_par["KIN"][:,0]/c, guess_dispersion=model_par["KIN"][:,1], base_order=1, quiet=True)
    indx_measurements = SpectralIndices.measure_indices(absdb=abs_db, bhddb=band_db, wave=manga_wave, flux=fluxes-em_model_flux, ivar=ivar, mask=None, redshift=model_par["KIN"][:,0]/c, bitmask=None)

    # Close all plots generated by the MaNGA DAP pipeline
    plt.close("all")
    plt.cla()
    plt.clf()

    return em_model_eml_par, indx_measurements


def save_lookup(spectra):

    """
    This function takes the output from the measure_spec function and turns the measurements, errors and masked values into a 
    useable format that are then saved as seperate compressed npz arrays.

    INPUTS:
        :spectra:
        This is an array of spectra output from the generate_spectra function. We do not recommend inputting large arrays of spectra at once.
        Instead, we recommend chunking this array with N = 10-100 spectra in each chunk. The fluxes should be reported at each manga wavelength, M 
        so that the shape is (N, M).

    OUTPUTS:
        :npz files:
        The compressed npz files that are saved will all have the shape (N, 7) - as there are 7 spectral parameters recorded for SNITCH. 
        If the files already exist, the measurements will be appended on to the end of the existing file. If starting a new run, remember to 
        either delete or rename any files from previous runs so that you know which values are relevant. 
    """

    eml, idm = measure_spec(spectra)
    emls = eml["EW"][:, np.where(emlines['name']=='Ha')].reshape(-1,1)
    idms =idm["INDX"].reshape(-1,8)

    emls_mask = eml["MASK"][:, np.where(emlines['name']=='Ha')].reshape(-1,1)
    idms_mask =idm["MASK"].reshape(-1,8)

    lu = np.append(emls, idms, axis=1).reshape(-1, 9) # Halpha 0th, Hbeta 2nd, MgB 3rd, Fe5270 4th, Fe5335 5th, HDeltaA 6th, D4000 7th
    lu_mask = np.append(emls_mask, idms_mask, axis=1).reshape(-1, 9) # Halpha 0th, Hbeta 2nd, MgB 3rd, Fe5270 4th, Fe5335 5th, HDeltaA 6th, D4000 7th, Dn4000 8th, Ti0 9th

    with l:
        if os.path.isfile("spectral_parameter_measurements_AA.npz"):
            with np.load("spectral_parameter_measurements_AA.npz") as sp_lu:
                np.savez("spectral_parameter_measurements_AA.npz", lookup=np.append(sp_lu["lookup"], lu, axis=0))
            with np.load("spectral_parameter_measurements_mask_AA.npz") as sp_lu_mask:
                np.savez("spectral_parameter_measurements_mask_AA.npz", lookupmask=np.append(sp_lu_mask["lookupmask"], lu_mask, axis=0))

        else:
            np.savez("spectral_parameter_measurements_AA.npz", lookup=lu)
            np.savez("spectral_parameter_measurements_mask_AA.npz", lookupmask=lu_mask)


def walker_plot(samples, nwalkers, ndim=3, limit=-1, truth=[np.nan, np.nan, np.nan]):

    """ Plotting function to visualise the steps of the walkers in each parameter dimension. 
        
        :samples:
        Array of shape (nsteps*nwalkers, 3) produced by the sample function in snitch_mcmc.py
        
        :nwalkers:
        The number of walkers that step around the parameter space used to produce the samples by the sample function. M
        ust be an even integer number larger than ndim.

        :ndim:
        Number of parameters fit by emcee. Default is 3. Optional. 
        
        :limit:
        Integer value less than nsteps to plot the walker steps to. Optional. 

        :truth:
        Actual values of [Z, tq, tau] if known - can also be used to input the 50th percentile to show the quoted values. Optional. 

        RETURNS:
        :fig:
        The figure object
        """
    s = samples.reshape(nwalkers, -1, ndim)
    s = s[:,:limit, :]
    fig = plt.figure(figsize=(8,12))
    ax1 = plt.subplot(3,1,1)
    ax2 = plt.subplot(3,1,2)
    ax3 = plt.subplot(3,1,3)
    ax1.plot(s[:,:,0].T, 'k')
    ax1.axhline(truth[0], color='r')
    ax2.plot(s[:,:,1].T, 'k')
    ax2.axhline(truth[1], color='r')
    ax3.plot(s[:,:,2].T, 'k')
    ax3.axhline(truth[2], color='r')
    ax1.tick_params(axis='x', labelbottom='off')
    ax2.tick_params(axis='x', labelbottom='off')
    ax3.set_xlabel(r'step number')
    ax1.set_ylabel(r'$Z$')
    ax2.set_ylabel(r'$t_{quench}$')
    ax3.set_ylabel(r'$\log_{10}$ $\tau$')
    plt.subplots_adjust(hspace=0.1)

    return fig
