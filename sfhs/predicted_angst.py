import sys, pickle, copy
import numpy as np
import matplotlib.pyplot as pl

import astropy.io.fits as pyfits
import fsps
from sedpy import observate
import regionsed as rsed
from lfutils import *
from sfhutils import load_angst_sfh


def total_galaxy_data(sfhfilename, zindex, filternames = None,
                      basti=False, lfstring=None, agb_dust=1.0):

    total_sfhs = load_angst_sfh(sfhfilename)
    zlist = [zindex]
    
    #########
    # Initialize the ingredients (SPS, SFHs, LFs)
    #########
    # SPS
    if filternames is not None:
        sps = fsps.StellarPopulation(add_agb_dust_model = True)
        sps.params['sfh'] = 0
        sps.params['agb_dust'] = agb_dust
        dust = ['nodust', 'agbdust']
        sps.params['imf_type'] = 0.0 #salpeter
        filterlist = observate.load_filters(filternames)
        zmets = [sps.zlegend[zindex-1]]
    else:
        filterlist = None
        
    #############
    # Sum the region SFHs into a total integrated SFH, and do the
    # temporal interpolations to generate integrated spectra, LFs, and
    # SEDs
    ############
    
    # Get LFs broken out by age and metallicity as well as the total
    # LFs. these are stored as a list of different metallicities
    bins = rsed.lfbins 
    if lfstring is not None:
        lffiles = [lfstring.format(z) for z in zlist]
        lf_base = [read_lfs(f) for f in lffiles]
        lfs_zt, lf, logages = rsed.one_region_lfs(copy.deepcopy(total_sfhs), lf_base)
    else:
        lfs_zt, lf, logages = None, None, None
    # Get spectrum and magnitudes
    if filterlist is not None:
        spec, wave, mass = rsed.one_region_sed(copy.deepcopy(total_sfhs), zmets, sps )
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
    
    filters = ['galex_NUV', 'spitzer_irac_ch1',
               'spitzer_irac_ch4', 'spitzer_mips_24']
    
    ldir, cdir = 'lf_data/', 'angst_composite_lfs/'
    # total_cloud_data will loop over the appropriate (for the
    # isochrone) metallicities for a given lfst filename template
    lfst = '{0}z{{0:02.0f}}_tau{1:2.1f}_vega_irac{2}_n2_teffcut_lf.txt'
    basti = False
    zindex, agb_dust = 1.0

    adir = ''
    galaxies=[]
    filenames = [adir + g for g in galaxies]
    for f in filenames:
        dat, sfhs = total_angst_data(f, zindex, filternames=filters, agb_dust=agb_dust,
                                     lfstring=lfstring, basti=basti)
        
