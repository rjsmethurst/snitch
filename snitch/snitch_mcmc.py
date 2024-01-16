import numpy as np
import emcee
import time
import os
from astropy.cosmology import Planck15

from itertools import product

from scipy import interpolate   

from george import kernels
import george
from scipy.interpolate import LinearNDInterpolator

######################################################################################################
#
# August 1st 2018, RJS
#
# This script runs the model optimisation step of SNITCH using the python code emcee written by 
# Daniel Foreman-Mackey --> http://dfm.io/emcee/current/
# 
# You will find that the previously generated look up table values are loaded and then a function 
# is set up to interpolate over these values. A user can change the interpolation routine how they wish
# and the lookup function for their needs. 
# 
# Similarly, lnprior, lnlikelihood and lnprob functions are defined which the user can change as they need
# for their science goals. 
#
# If you have any questions please email rjsmethurst@gmail.com
#
# July 18th 2019
#
# NearestND isn't working in reality with MPL6 AGN - can we change to use Gaussian Regression as the interpolator? 
#
######################################################################################################


pad_zmet = np.array([0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.001, 0.0012, 0.0016, 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.012, 0.015, 0.019, 0.024, 0.03])
pad_zmetsol = 0.019

pad_solmet = pad_zmet/pad_zmetsol
zmets = np.append((np.linspace(0, 10, 11)*2 + 1).astype(int), [22.])
zsolmets = pad_solmet[(zmets-1).astype(int)] 

#ages = Planck15.age(10**np.linspace(-0.824, -2.268, 25))[-7:-5].value
time_steps = Planck15.age(10**np.linspace(-0.824, -3.295, 15)).reshape(-1,1,1).value
taus = 10**np.linspace(6, 9.778, 50)/1e9

with np.load('emls_par_pool_mapped_nozshift_ppxfcorrect_AA_12zmet.npz') as orig_pred:
    pred = orig_pred['lookup']
    
with np.load('emls_mask_par_pool_mapped_nozshift_ppxfcorrect_AA_12zmet.npz') as orig_mask:
    mask = orig_mask['lookupmask']
    
tqs = np.append(np.flip(time_steps.flatten()[0]- 10**(np.linspace(7, np.log10((time_steps.flatten()[0]-0.1)*1e9), 48))/1e9, axis=0), [time_steps.flatten()[0]-0.001, time_steps.flatten()[0]+0.1], axis=0)

sv = np.array(list(product(time_steps[0][0], tqs, np.log10(taus), zsolmets)))
for n in range(1, len(time_steps)):
    tqs = np.append(np.flip(time_steps.flatten()[n]- 10**(np.linspace(7, np.log10((time_steps.flatten()[n]-0.1)*1e9), 48))/1e9, axis=0), [time_steps.flatten()[n]-0.001, time_steps.flatten()[n]+0.1], axis=0)
    sv = np.append(sv, np.array(list(product(time_steps[n][0], tqs, np.log10(taus), zsolmets))), axis=0)
    

masked_sp = np.ma.masked_array(data=pred, mask=mask)#.reshape(15, 50, 50, 12, -1)

# Create the interpolation function which will allow us to calcualte the spectral paramter values at any [t_obs, tq, tau, Z]
deltq = sv[420000:, 0] - sv[420000:,1]
newsv = sv[420000:, 1:]
newsv[:,0] = deltq

new_masked_sp = masked_sp[420000:, :]
f = LinearNDInterpolator(newsv, new_masked_sp)

#f = interpolate.NearestNDInterpolator(sv, masked_sp)
#ker = kernels.ExpSquaredKernel(0.5, ndim=4)

def lookup(theta, age, func):
    """
    Predict the values of the model spectral parameters using the look up table for the test SFH parameters. 
    
    INPUTS
    :theta:
    An array of size (3,) containing the test values [deltatq, tau, Z].

    :age:
    Observed age of a galaxy, often calculated from the redshift i.e. at z=0.1 the age ~ 12.5. Must be in units of Gyr. An array of shape (1,).
    
    RETURNS:
    The predicted spectral parameter values calculated using interpolation over the look up table. 
    """
    pred = f([theta[0], theta[1], theta[2]])[0]


    ha_pred = pred[0]
    d4000_pred = pred[7]
    hb_pred = pred[1]
    hdA_pred = pred[5]
    mgfe_pred = np.sqrt( pred[2] * ( 0.72*pred[3] + 0.28*pred[4] )  )  
    # print(np.array([age, theta[1], theta[2], theta[0]]).reshape(1, 4))


    return ha_pred, d4000_pred, hb_pred, hdA_pred, mgfe_pred

    
def lnlikelihood(theta, ha, e_ha, d4000, e_d4000, hb, e_hb, hdA, e_hdA, mgfe, e_mgfe, age, func):
    """ 
    Function for determining the logarithmic likelihood for a given [Z, tq, tau].
    
    :theta:
    An array of size (3,) containing the test values [Z, tq, tau].

    :ha, d4000, hb, hdA, mgfe:
    Spectral parameter measurements for the spectra the user is trying to fit a SFH to - note that there are 5 measurements. 
    They do not necessarily have to match the names but the order has to match whatever lookup returns from the look up table. 

    :e_ha, e_d4000, e_hb, e_hdA, e_mgfe:
    Same as above but for the measurement error on each spectral parameter. 

    :age:
    Observed age of a galaxy, often calculated from the redshift i.e. at z=0.1 the age ~ 12.5. Must be in units of Gyr. An array of shape (1,).
    
    
    RETURNS:
    Array containing the likelihood for each galaxy at the given :theta:
    """

    ha_pred, d4000_pred, hb_pred, hdA_pred, mgfe_pred = lookup(theta, age, func)
    return np.nansum([-0.5*np.log(2*np.pi*e_ha**2)-0.5*((ha-ha_pred)**2/e_ha**2) , -0.5*np.log(2*np.pi*e_d4000**2)-0.5*((d4000-d4000_pred)**2/e_d4000**2), -0.5*np.log(2*np.pi*e_hb**2)-0.5*((hb-hb_pred)**2/e_hb**2), -0.5*np.log(2*np.pi*e_hdA**2)-0.5*((hdA-hdA_pred)**2/e_hdA**2), -0.5*np.log(2*np.pi*e_mgfe**2)-0.5*((mgfe-mgfe_pred)**2/e_mgfe**2)])

# Overall likelihood function combining prior and model
def lnprobability(theta, ha, e_ha, d4000, e_d4000, hb, e_hb, hdA, e_hdA, mgfe, e_mgfe, age, func):
    """
    Overall logarithmic posterior function combining the prior and calculating the likelihood. 
        
        :theta:
        An array of size (3,) containing the test values [Z, tq, tau].

        :ha, d4000, hb, hdA, mgfe:
        Spectral parameter measurements for the spectra the user is trying to fit a SFH to - note that there are 5 measurements. 
        They do not necessarily have to match the names but the order has to match whatever lookup returns from the look up table. 

        :e_ha, e_d4000, e_hb, e_hdA, e_mgfe:
        Same as above but for the measurement error on each spectral parameter. 
   
        :age:
        Observed age of a galaxy, often calculated from the redshift i.e. at z=0.1 the age ~ 12.5. Must be in units of Gyr. An array of shape (1,).
        
        RETURNS:
        Value of the posterior function for the given :theta: value.
        
        """
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlikelihood(theta, ha, e_ha, d4000, e_d4000, hb, e_hb, hdA, e_hdA, mgfe, e_mgfe, age, func)

def lnprior(theta):
    """ 
    Function to calcualte the logarithmic prior on theta values of the Z, tq and tau parameters. Defined ranges are specified 
    - outside these ranges the function returns -np.inf and does not calculate the posterior probability. 
       
    INPUTS 
        :theta: 
        An array of size (3,) containing the test values [dtq, tau, Z].

    RETURNS:
        Value of the prior at the specified :theta: value.
        """
    dtq, tau, Z = theta
    if -0.1 <= dtq <= 13.807108309208775 and np.log10(0.00001) <= tau <= np.log10(5.9) and 0.001 <= Z <= 1.6:
        return 0.0
    elif -0.1 <= dtq <= 13.807108309208775 and np.log10(5.9) < tau <= np.log10(6.0) and 0.001 <= Z <= 1.6:
        return 2*(np.exp(np.log10(5.9)) - np.exp(np.log10(tau)))
    else:
        return -np.inf

def sample(path, ndim=3, nwalkers=100, nsteps=100, burnin=500, start=[1.0, -0.7, 1.0], ha=np.nan, e_ha=np.nan, d4000=np.nan, e_d4000=np.nan, hb=np.nan, e_hb=np.nan, hdA=np.nan, e_hdA=np.nan, mgfe=np.nan, e_mgfe=np.nan, age=Planck15.age(0).value, ID=0):
    """ 
    Function to implement the emcee EnsembleSampler function for the sample of galaxies input. Burn in is run and calcualted fir the length specified before the sampler is reset and then run for the length of steps specified. 
        
    INPUTS
        :path:
        Directory path which output should be saved to 

        :ndim:
        The number of parameters in the model that emcee must find. In the default expsfh case it always 3 with Z, tq, tau.
        
        :nwalkers:
        The number of walkers that step around the parameter space. Must be an even integer number larger than ndim. 
        
        :nsteps:
        The number of steps to take in the final run of the MCMC sampler. Integer.
        
        :burnin:
        The number of steps to take in the inital burn-in run of the MCMC sampler. Integer. 
        
        :start:
        The positions in the Z, tq and tau parameter space to start. An array of shape (1,3).
         
        :ha, d4000, hb, hdA, mgfe:
        Spectral parameter measurements for the spectra the user is trying to fit a SFH to - note that there are 5 measurements. 
        They do not necessarily have to match the names but the order has to match whatever lookup returns from the look up table. 

        :e_ha, e_d4000, e_hb, e_hdA, e_mgfe:
        Same as above but for the measurement error on each spectral parameter. 

        :age:
        Observed age of a galaxy, often calculated from the redshift i.e. at z=0.1 the age ~ 12.5. Must be in units of Gyr. An array of shape (1,).
        
        :id:
        ID number to specify which galaxy this run is for.
        
    RETURNS:
        :samples:
        Array of shape (nsteps*nwalkers, ndim) containing the positions of the walkers at all steps for each SFH parameter in ndim.
        
        """

    print('emcee running...')
    p0 = [start + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    # idx = np.searchsorted(time_steps.flatten(), age)
    # if idx == len(time_steps.flatten()):
    #     idx = len(time_steps.flatten())-1
    # else:
    #     pass
    func = np.sin
    # # newmasked_sp = masked_sp[idx,:,:,:,:]
    # # func = interpolate.RegularGridInterpolator((tqs, np.log10(taus), zsolmets), newmasked_sp, method='linear', bounds_error=False, fill_value=np.nan)    
    # newsv = sv[30000*(idx):30000*(idx+1)][:,1:]
    # newmasked_sp = masked_sp[idx,:,:,:,:]
    # #func = interpolate.NearestNDInterpolator(newsv, newmasked_sp.reshape(-1, 9))
    # func = interpolate.NearestNDInterpolator(newsv, newmasked_sp.reshape(-1, 9))
    #func = interpolate.RegularGridInterpolator((tqs, np.log10(taus), zsolmets), newmasked_sp, method='linear', bounds_error=False, fill_value=np.nan)    
    #gp_hodlr = george.GP(ker, solver=george.HODLRSolver, seed=42)
    #gp_hodlr.compute(sv[30000*(idx-1):30000*(idx+1)])    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobability, args=(ha, e_ha, d4000, e_d4000, hb, e_hb, hdA, e_hdA, mgfe, e_mgfe, age, func))
    """ Burn in run here..."""
    pos, prob, state = sampler.run_mcmc(p0, burnin)
    lnp = sampler.flatlnprobability
    np.savez(path+'/lnprob_burnin_'+str(ID)+'.npz', lnp=lnp)
    samples = sampler.chain[:,:,:].reshape((-1,ndim))
    samples_save = path+'/samples_burn_in_'+str(ID)+'.npz'
    np.savez(samples_save, samples=samples)
    sampler.reset()
    print('Burn in complete...')
    """ Main sampler run here..."""
    sampler.run_mcmc(pos, nsteps)
    lnpr = sampler.flatlnprobability
    np.savez(path+'/lnprob_run_'+str(ID)+'.npz', lnp=lnpr)
    samples = sampler.chain[:,:,:].reshape((-1,ndim))
    samples_save = path+'/samples_'+str(ID)+'.npz'
    np.savez(samples_save, samples=samples)
    print('Main emcee run completed.')
    sampler.reset()

    return samples


