from snitch_mcmc import *
from functions import walker_plot
from astropy.cosmology import Planck15
import numpy as np
import sys
import corner
import matplotlib.pyplot as plt
import sys
import os
from scipy.optimize import basinhopping
from scipy import interpolate

from argparse import ArgumentParser


import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (10000,-1))

np.set_printoptions(suppress=True, precision=4)

######################################################################################################
#
# August 1st 2018, RJS
#
# This script takes emission and absorption spectral features from a single spectra and returns the best 
# best fit parameters of an exponentially declining SFH, [Z, tq, tau] to describe that spectra. 
# 
# A user can alter this script in order to load a file with measured spectral parameters 
# 
# Similarly, lnprior, lnlikelihood and lnprob functions are defined which the user can change as they need
# for their science goals. 
#
# If you have any questions please email rjsmethurst@gmail.com
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

with np.load('/Users/smethurst/Projects/mangaagn/snitch/snitch/emls_par_pool_mapped_nozshift_ppxfcorrect_AA_12zmet.npz') as orig_pred:
    pred = orig_pred['lookup']
    
with np.load('/Users/smethurst/Projects/mangaagn/snitch/snitch/emls_mask_par_pool_mapped_nozshift_ppxfcorrect_AA_12zmet.npz') as orig_mask:
    mask = orig_mask['lookupmask']
    
tqs = np.append(np.flip(time_steps.flatten()[0]- 10**(np.linspace(7, np.log10((time_steps.flatten()[0]-0.1)*1e9), 48))/1e9, axis=0), [time_steps.flatten()[0]-0.001, time_steps.flatten()[0]+0.1], axis=0)

sv = np.array(list(product(time_steps[0][0], tqs, np.log10(taus), zsolmets)))
for n in range(1, len(time_steps)):
    tqs = np.append(np.flip(time_steps.flatten()[n]- 10**(np.linspace(7, np.log10((time_steps.flatten()[n]-0.1)*1e9), 48))/1e9, axis=0), [time_steps.flatten()[n]-0.001, time_steps.flatten()[n]+0.1], axis=0)
    sv = np.append(sv, np.array(list(product(time_steps[n][0], tqs, np.log10(taus), zsolmets))), axis=0)
    

masked_sp = np.ma.masked_array(data=pred, mask=mask).reshape(15, 50, 50, 12, -1)


def snitch(ha, e_ha, d4000, e_d4000, hb, e_hb, hdA, e_hdA, mgfe, e_mgfe, redshift, ident, opstart= [1.0, -1.0, 1.0]):

    age = Planck15.age(redshift).value
    nll = lambda *args: -lnprobability(*args)

    nwalkers = 100 # number of monte carlo chains
    nsteps= 200 # number of steps in the monte carlo chain
    burnin = 1000 # number of steps in the burn in phase of the monte carlo chain
    ndim = 3 # number of dimensions in the SFH model

    # idx = np.searchsorted(time_steps.flatten(), age)
    # if idx == len(time_steps.flatten()):
    #     idx = len(time_steps.flatten())-1
    # else:
    #     pass
    # # newmasked_sp = masked_sp[idx,:,:,:,:]
    # # func = interpolate.RegularGridInterpolator((tqs, np.log10(taus), zsolmets), newmasked_sp, method='linear', bounds_error=False, fill_value=np.nan)    
    # newsv = sv[30000*(idx):30000*(idx+1)][:,1:]
    # newmasked_sp = masked_sp[idx,:,:,:,:]
    # func = interpolate.NearestNDInterpolator(newsv, newmasked_sp.reshape(-1, 9))
    func = np.sin

    result_bh = basinhopping(nll, opstart, minimizer_kwargs={"args": (ha, e_ha, d4000, e_d4000, hb, e_hb, hdA, e_hdA, mgfe, e_mgfe, age, func), "method":'Nelder-Mead'})
    print(result_bh)
    if "successfully" in result_bh.message[0]:
       start = result_bh['x']
    else:
       start = np.array(opstart)

    #The rest of this file calls the emcee module which is initialised in the sample function of the posterior file. 
    samples = sample(path=os.getcwd(), ndim=ndim, nwalkers=nwalkers, nsteps=nsteps, burnin=burnin, start=start, ha=ha, e_ha=e_ha, d4000=d4000, e_d4000=e_d4000, hb=hb, e_hb=e_hb, hdA=hdA, e_hdA=e_hdA, mgfe=mgfe, e_mgfe=e_mgfe, age=age, ID=ident)

    # This section of the code prunes the walker positions returned by emcee to remove those stuck in local minima. We follow the method
    # outlined in Hou et al. (2012).
    with np.load('lnprob_run_'+str(ident)+'.npz') as lnp:
        lk = np.mean(lnp['lnp'].reshape(nwalkers, nsteps), axis=1)
        idxs = np.argsort(-lk)
        slk = -lk[idxs]
        cluster_idx = np.argmax(np.diff(slk) >  10000*np.diff(slk)[0]/ (np.linspace(1, len(slk)-1, len(slk)-1)-1))+1
        if cluster_idx > 1:
            #lnps = slk[:cluster_idx]
            samples = samples.reshape(nwalkers, nsteps, ndim)[idxs,:,:][:cluster_idx,:,:].reshape(-1,ndim)
        else:
            pass
        lnp.close()
        del lnp, lk, idxs, slk, cluster_idx 
                
    dtq_mcmc, log_tau_mcmc, Z_mcmc,  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16,50,84], axis=0)))

    # Save the inferred SFH parameters. In each case the elements are [best fit value, plus uncertainty,  minus uncertainty].
    # Note that the log tau values are recorded. 
    np.save('inferred_SFH_parameters_ID_'+str(ident)+'.npy', [dtq_mcmc, log_tau_mcmc, Z_mcmc])
    
    # Produce the emcee corner plot showing which part of the parameter space the walkers explored. 

    try:
        fig = corner.corner(samples, labels=[r'$\delta t_q$', r'$\log_{10}\tau$', r'Z'], quantiles=([0.16, 0.5, 0.84]))    
        fig.savefig('snitch_output_corner_'+str(ident)+'.pdf')
        plt.close(fig)
    except(ValueError):
        pass

    ### The lines below produce the walker positions with each step for the burn in phase and the rest of the run.
    ### Uncomment this section if you'd like these produced. 

    try:
        fig = walker_plot(samples, nwalkers, ndim, -1, [dtq_mcmc[0], log_tau_mcmc[0], Z_mcmc[0]])
        fig.tight_layout()
        fig.savefig('walkers_steps_with_pruning_'+str(ident)+'.pdf')
        plt.close(fig)
    except(ValueError):
        pass

    with np.load('samples_burn_in_'+str(ident)+'.npz') as burninload:
        try:
            fig = walker_plot(burninload['samples'], nwalkers, ndim, -1, [dtq_mcmc[0], log_tau_mcmc[0], Z_mcmc[0]])
            fig.tight_layout()
            fig.savefig('walkers_steps_burn_in_without_pruning_'+str(ident)+'.pdf')
            plt.close(fig)
        except(ValueError):
            pass
        burninload.close()

    plt.close('all')

    # Print out the best fit values. Note that the actual value of tau in Gyr is printed, not the log value. 

    print(r'Best fit Z value (3.s.f.) found by SNITCH for', ident, 'input parameters are : [ {0:1.3f}, +{1:1.3f}, -{2:1.3f} ]'.format(Z_mcmc[0], Z_mcmc[1], Z_mcmc[2]))
    print(r'Best fit dt_q value (3.s.f.) found by SNITCH for', ident, 'input parameters are : [ {0:1.3f}, +{1:1.3f}, -{2:1.3f} ]'.format(dtq_mcmc[0], dtq_mcmc[1], dtq_mcmc[2]))
    print(r'Best fit tau value (3.s.f.) found by SNITCH for', ident, 'input parameters are : [ {0:1.3f}, +{1:1.3f}, -{2:1.3f} ]'.format(10**log_tau_mcmc[0], 10**(log_tau_mcmc[1]+log_tau_mcmc[0])-10**log_tau_mcmc[0], 10**log_tau_mcmc[0] - 10**(log_tau_mcmc[0]-log_tau_mcmc[2])))
    return(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16,50,84], axis=0))))

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('ha', type=float, help=r'EW[H$\alpha$]')
    parser.add_argument('e_ha', type=float, help=r'measurement unertainty on EW[H$\alpha$]')
    parser.add_argument('d4000', type=float, help=r'D4000')
    parser.add_argument('e_d4000', type=float, help=r'measurement unertainty on D4000')
    parser.add_argument('hb', type=float, help=r'H$\beta$ absorption index')
    parser.add_argument('e_hb', type=float, help=r'measurement unertainty on H$\beta$ absorption index')
    parser.add_argument('hdA', type=float, help=r'H$\delta_A$ absorption index')
    parser.add_argument('e_hdA', type=float, help=r'measurement unertainty on H$\delta_A$ absorption index')
    parser.add_argument('mgfe', type=float, help=r"MgFe'")
    parser.add_argument('e_mgfe', type=float, help=r"measurement unertainty on MgFe'")
    parser.add_argument('redshift', type=float, help=r'Redshift of the spectrum')
    parser.add_argument('ident', type=str, help=r'Identification number of the spectrum')


    arg = parser.parse_args()

    # Define parameters needed for emcee - these can be changed depending on the users requirements. 

    nwalkers = 100 # number of monte carlo chains
    nsteps= 200 # number of steps in the monte carlo chain
    opstart = [1.0, -1.0, 1.0] # starting place for the scipy optimisation chains [deltatq, tau, Z]
    burnin = 1000 # number of steps in the burn in phase of the monte carlo chain
    ndim = 3 # number of dimensions in the SFH model


    nll = lambda *args: -lnprobability(*args)

    dtq_mcmc, tau_mcmc, Z_mcmc = snitch(arg.ha, arg.e_ha, arg.d4000, arg.e_d4000, arg.hb, arg.e_hb, arg.hdA, arg.e_hdA, arg.mgfe, arg.e_mgfe, arg.redshift, arg.ident)






