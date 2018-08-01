from snitch_mcmc import *
from astropy.cosmology import Planck15
import numpy as np
import sys
import corner
import matplotlib.pyplot as plt
import sys
import os
from scipy.optimize import basinhopping

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

def snitch(ha, e_ha, d4000, e_d4000, hb, e_hb, hdA, e_hdA, mgfe, e_mgfe, redshift, ident):

    age = Planck15.age(redshift).value

    result_bh = basinhopping(nll, opstart, minimizer_kwargs={"args": (ha, e_ha, d4000, e_d4000, hb, e_hb, hdA, e_hdA, mgfe, e_mgfe, age), "method":'Nelder-Mead'})
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
                
    Z_mcmc, tq_mcmc, log_tau_mcmc,  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16,50,84], axis=0)))

    # Save the inferred SFH parameters. In each case the elements are [best fit value, plus uncertainty,  minus uncertainty].
    # Note that the log tau values are recorded. 
    np.save('inferred_SFH_parameters_ID_'+str(ident)+'.npy', [Z_mcmc, tq_mcmc, log_tau_mcmc])
    
    # Produce the emcee corner plot showing which part of the parameter space the walkers explored. 

    fig = corner.corner(samples, labels=[r'$Z$', r'$t_q$', r'$\log_{10}\tau$'], quantiles=([0.16, 0.5, 0.84]))    
    fig.savefig('starpy_output_corner_testing_no_oii_finer_pruning_'+str(n)+'newtqs_random_values.png')
    plt.close(fig)

    ### The lines below produce the walker positions with each step for the burn in phase and the rest of the run.
    ### Uncomment this section if you'd like these produced. 

    # try:
    #     fig = walker_plot(samples, nwalkers, -1, [k[n,0], k[n,1], k[n,2]], n)
    #     fig.tight_layout()
    #     fig.savefig('walkers_steps_pruning_no_oii_finer_'+str(n)+'newtqs_random_values.pdf')
    #     plt.close(fig)
    # except(ValueError):
    #     pass

    # with np.load('samples_burn_in_'+str(n)+'.npz') as burninload:
    #     try:
    #         fig = walker_plot(burninload['samples'], nwalkers, -1, [k[n,0], k[n,1], k[n,2]], n)
    #         fig.tight_layout()
    #         fig.savefig('walkers_steps_burn_in_wihtout_pruning_no_oii_finer_'+str(n)+'newtqs_random_values.pdf')
    #         plt.close(fig)
    #     except(ValueError):
    #         pass
    #     burninload.close()

    plt.close('all')

    # Print out the best fit values. Note that the actual value of tau in Gyr is printed, not the log value. 

    print('Best fit [Z, tq, tau] values found by SNITCH for input parameters are : [', Z_mcmc[0], tq_mcmc[0], 10**log_tau_mcmc[0], ']')


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
    opstart = [1.0, 12.0, np.log10(0.25)] # starting place for the scipy optimisation chains
    burnin = 1000 # number of steps in the burn in phase of the monte carlo chain
    ndim = 3 # number of dimensions in the SFH model


    nll = lambda *args: -lnprobability(*args)

    snitch(arg.ha, arg.e_ha, arg.d4000, arg.e_d4000, arg.hb, arg.e_hb, arg.hdA, arg.e_hdA, arg.mgfe, arg.e_mgfe, arg.redshift, arg.ident)






