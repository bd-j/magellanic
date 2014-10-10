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

#Define SPS object as global
sps = fsps.StellarPopulation(add_agb_dust_model=True)
sps.params['sfh'] = 0
sps.params['imf_type'] = 0.0 #salpeter

def zsfh_to_obs(sfhlist, zlist, lfbands=None, select_function=None,
                filternames=None,
                basti=False, agb_dust=1.0):
    
    sps.params['agb_dust'] = agb_dust
    nsfh = len(sfhlist)
    assert len(zlist) == nsfh
        
    #############
    # LFs
    ############
    #create the SSP CLFs
    lf_base = []
    bins = rsed.lfbins
    for zmet in zlist:
        sps.params['zmet'] = np.abs(sps.zlegend - zmet).argmin() + 1
        lf_base += [isochrone_to_clf(sps, lfbands, bins,
                                     select_function=select_function)]
    #use the SSP CLFs to generate a total LF
    lfs_zt, lf, logages = rsed.one_region_lfs(copy.deepcopy(total_sfhs), lf_base)
    
    ###########
    # SED
    ############
    if filternames is not None:
        filterlist = observate.load_filters(filternames)
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

def sum_sfhs(sfhs1, sfhs2):
    """
    Accumulate individual sets of SFHs into a total set of SFHs.  This
    assumes that the individual SFH sets all have the same number and
    order of metallicities, and the same time binning.
    """
    if sfhs1 is None:
        return copy.deepcopy(sfhs2)
    elif sfhs2 is None:
        return copy.deepcopy(sfhs1)
    else:
        out = copy.deepcopy(sfhs1)
        for s1, s2 in zip(out, sfhs2):
            s1['sfr'] += s2['sfr']
        return out
    
if __name__ == '__main__':
    
    filters = ['galex_NUV', 'spitzer_irac_ch2',
               'spitzer_irac_ch4', 'spitzer_mips_24']
    #filters = None
    
    basti = False
    agb_dust=1.0
    agebins = np.arange(9)*0.3 + 7.4
    
    #loop over clouds (and bands and agb_dust) to produce clfs
    for cloud in ['smc']:
        sfhs = could_data(cloud)
