import sys, pickle, copy
import numpy as np
import matplotlib.pyplot as pl

import astropy.io.fits as pyfits
import regionsed as rsed
import mcutils as utils
from lfutils import *
import fsps
try:
    from sedpy import observate
except ImportError:
    # you wont be able to predict the magnitudes
    # filterlist must be set to None in calls to total_cloud_data


def zsfh_to_obs(sfhlist, zlist, lfbandnames=None, select_function=None,
                bandnames=None, sps=None, isocs=None):
    
    #basti = np.any(sps.zlegend[0:2] == 0.0006) #Hack to check for Basti Isochrones
    nsfh = len(sfhlist)
    assert len(zlist) == nsfh
        
    #############
    # LFs
    ############
    #create the SSP CLFs
    lf_base = []
    bins = rsed.lfbins
    for i,zmet in enumerate(zlist):
        if isocs is None:
            sps.params['zmet'] = np.abs(sps.zlegend - zmet).argmin() + 1
            isoc = sps.cmd()
        else:
            isoc = isocs[i]
        lf_base += [isochrone_to_clf(isoc, lfbandnames, bins,
                                     select_function=select_function)]
    #use the SSP CLFs to generate a total LF
    lfs_zt, lf, logages = rsed.one_region_lfs(copy.deepcopy(total_sfhs), lf_base)
    
    ###########
    # SED
    ############
    if bandnames is not None:
        filterlist = observate.load_filters(bandnames)
        spec, wave, mass = rsed.one_region_sed(copy.deepcopy(total_sfhs), total_zmet, sps)
        mags = observate.getSED(wave, spec*rsed.to_cgs, filterlist=filterlist)
        maggies = 10**(-0.4 * np.atleast_1d(mags))
    else:
        maggies, mass = None, None
        
    #############
    # Write output
    ############
    total_values = {}
    total_values['agb_clf'] = lf
    total_values['agb_clfs_zt'] = lfs_zt
    total_values['clf_mags'] = bins
    total_values['logages'] = logages
    total_values['sed_ab_maggies'] = maggies
    total_values['sed_filters'] = filternames
    total_values['lffile'] = lfstring
    total_values['mstar'] = mass
    total_values['zlist'] = zlist
    return total_values, total_sfhs
    
if __name__ == '__main__':
    
    filters = ['galex_NUV', 'spitzer_irac_ch2',
               'spitzer_irac_ch4', 'spitzer_mips_24']
    #filters = None
    #Define SPS object
    sps = fsps.StellarPopulation(add_agb_dust_model=True,
                                 tpagb_norm_type=1)
    sps.params['sfh'] = 0
    sps.params['imf_type'] = 0.0 #salpeter
    sps.params['agb_dust'] = 1.0
    agebins = np.arange(9)*0.3 + 7.4

    regions = utils.smc_regions()
    if 'header' in regions.keys():
        rheader = regions.pop('header') #dump the header info from the reg. dict        
    total_sfhs = None
    for n, dat in regions.iteritems():
        total_sfhs = sum_sfhs(total_sfhs, dat['sfhs'])
        total_zmet = dat['zmet']

