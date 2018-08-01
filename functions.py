import os
import numpy as np

from argparse import ArgumentParser
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from astropy.cosmology import Planck15 

from itertools import product
from tqdm import tqdm, trange

import fsps

from mangadap.util.instrument import spectrum_velocity_scale

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

from astropy import constants as con
from astropy import units as un
c = con.c.to(un.km/un.s).value


def sfh(tq, tau, time):
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



def generate_spectra(sfr, ages, time_steps, zmets):

    """ 
    This function allows the user to generate many spectra for a defined SFR at given ages, at user defined time steps and metallicities. 

    INPUT:
        :sfr:
        The SFRs defined at each age for any number of sfhs, H. Shape (H, SFR)
        
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

    sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=0, zmet=18, sfh=3, dust_type=2, dust2=0.2, 
        add_neb_emission=True, add_neb_continuum=False, imf_type=1, add_dust_emission=True, min_wave_smooth=3600, max_wave_smooth=7500, smooth_velocity=True, sigma_smooth=77.)

    sp.set_tabular_sfh(age = ages, sfr=sfr)

    def time_spec(params):
        sps = sp.get_spectrum(tage=params[0], zmet=params[1])[1]
        return sps

    fsps_wave = sp.get_spectrum()[0]
    fsps_spec = np.array(list(map(time_spec, list(product(time_steps, zmets)))))


    f = interpolate.interp1d(fsps_wave, fsps_spec)
    fluxes = f(manga_wave)

    return fsps_flux 


def measure_spec(fluxes):
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
                              file_path='~/extindxsnitch.par')
    abs_db = AbsorptionIndexDB(u"USERABS", indxdb_list=define_abs_db)

    band_db = BandheadIndexDB(u"BHBASIC")
    global indx_names
    indx_names = np.hstack([abs_db.data["name"], band_db.data["name"]])

    # Define the emission line feature database
    specm = EmissionLineModelBitMask()
    elric = Elric(specm)
    global emlines
    define_em_db = SpectralFeatureDBDef(key='USEREM',
                              file_path='~/elpsnitch.par')
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
    model_wave, qmodel_flux, model_mask, model_par  = ppxf.fit(tpl["WAVE"].data.copy(), tpl["FLUX"].data.copy(), fsps_wave, fluxes, np.ones_like(fluxes), guess_redshift, guess_dispersion, iteration_mode="none", velscale_ratio=1, degree=8, mdegree=-1, moments=2, quiet=True)
    em_model_wave, qem_model_flux, qem_model_base, em_model_mask, em_model_fit_par, em_model_eml_par = elric.fit(fsps_wave, fluxes, emission_lines=emlines, ivar=np.ones_like(fluxes), sres=np.ones_like(fluxes), continuum=model_flux, guess_redshift = model_par["KIN"][:,0]/c, guess_dispersion=model_par["KIN"][:,1], base_order=1, quiet=True)
    indx_measurements = SpectralIndices.measure_indices(absdb=abs_db, bhddb=band_db, wave=fsps_wave, flux=fluxes, ivar=np.ones_like(fluxes), mask=None, redshift=model_par["KIN"][:,0]/c, bitmask=None)

    # Close all plots generated by the MaNGA DAP pipeline
    plt.close("all")
    plt.cla()
    plt.clf()

    return em_model_eml_par, indx_measurement



