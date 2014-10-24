import sys, pickle, copy
import numpy as np
import matplotlib.pyplot as pl

import astropy.io.fits as pyfits
import regionsed as rsed
from lfutils import *
import fsps
try:
    from sedpy import observate
except ImportError:
    # you wont be able to predict the magnitudes
    # filterlist must be set to None in calls to total_cloud_data
    pass

   
def zsfh_to_obs(sfhlist, zlist, lfbandnames=None, select_function=None,
                bandnames=None, sps=None, isocs=None):
    """
    Go from a list of SFHs (one for each metallicity) to a broadband SED and set of
    luminosity functions for a stellar population.
    """
    sed_values, lf_values = {}, []
    #basti = np.any(sps.zlegend[0:2] == 0.0006) #Hack to check for Basti Isochrones
    nsfh = len(sfhlist)
    assert len(zlist) == nsfh

    ###########
    # SED
    ############
    if bandnames is not None:
        filterlist = observate.load_filters(bandnames)
        spec, wave, mass = rsed.one_region_sed(copy.deepcopy(total_sfhs), total_zmet, sps)
        mags = observate.getSED(wave, spec*rsed.to_cgs, filterlist=filterlist)
        maggies = 10**(-0.4 * np.atleast_1d(mags))
        sed_values['sed_ab_maggies'] = maggies
        sed_values['sed_filters'] = bandnames

    #############
    # LFs
    ############
    #create the SSP CLFs, using nearest neighbor interpolation for the metallicity
    all_lf_base = []
    bins = rsed.lfbins
    for i,zmet in enumerate(zlist):
        if isocs is not None:
            isoc = isocs[i]
        else:
            sps.params['zmet'] = np.abs(sps.zlegend - zmet).argmin() + 1
            isoc = sps.isochrones()
            print("Using Zmet={0} in place of requested "
            "Zmet={1}".format(sps.zlegend[sps.params['zmet']+1],zmet))

        ldat = isochrone_to_clfs(copy.deepcopy(isoc), lfbandnames,
                                 select_function=select_function)
        all_lf_base += [ldat]
    #use the SSP CLFs to generate a total LF (for each band)
    for i, band in enumerate(lfbandnames):
        lf_oneband = {}
        lf_base = [zbase[i] for zbase in all_lf_base]
        lfs_zt, lf, logages = rsed.one_region_lfs(copy.deepcopy(total_sfhs), lf_base)
        lf_oneband['bandname'] = band
        lf_oneband['clf'] = lf
        lf_oneband['clf_mags'] = bins
        lf_oneband['logages'] = logages
        lf_oneband['clfs_zt'] = lfs_zt
        
        lf_values += [lf_oneband]
        
    #############
    # Write output
    ############
    return sed_values, lf_values
    
if __name__ == '__main__':
    
    def select_function(isoc_dat, isoc_hdr):
        """
        Here's a function that selects certain rows from the full
        isochrone data and returns them.  The selection criteria can
        involve any of the columns given by the isochrone data, including
        magnitudes (or colors) as well as things like logg, logL, etc.
        """
        #select only objects cooler than 4000K and in tp-agb phase
        select = ( (isoc_dat['logt'] < np.log10(4000.0)) &
                   (isoc_dat['phase'] == 5.0)
                   )        
        return isoc_dat[select]

    # These are the filters for integrated magnitudes of the object
    sedfilters = ['galex_NUV', 'spitzer_irac_ch2',
                  'spitzer_irac_ch4', 'spitzer_mips_24']
    sedfilters = None
    # These are the filters for which you want LFs
    lffilters = ['2mass_ks','irac_2','irac_4']

    ########
    # Get the (smc) SFH
    ########
    import mcutils
    dm=18.9
    regions = mcutils.smc_regions()
    if 'header' in regions.keys():
        rheader = regions.pop('header') #dump the header info from the reg. dict        
    total_sfhs = None
    for n, dat in regions.iteritems():
        total_sfhs = mcutils.sum_sfhs(total_sfhs, dat['sfhs'])
        total_zmet = dat['zmet']

    #######
    #Define SPS object
    #######
    sps = fsps.StellarPopulation(add_agb_dust_model=True,
                                 tpagb_norm_type=2,
                                 compute_vega_mags=True)
    sps.params['sfh'] = 0
    sps.params['imf_type'] = 0.0 #salpeter
    sps.params['agb_dust'] = 1.0
    agebins = np.arange(9)*0.3 + 7.4

    # Go from SFH to LFs
    sed, clfs = zsfh_to_obs(total_sfhs, total_zmet,
                            lfbandnames=lffilters,
                            bandnames=sedfilters, sps=sps,
                            select_function=select_function) 

    pl.figure()
    for lf in clfs:
        pl.plot(lf['clf_mags']+dm, lf['clf'], label=lf['bandname'])
    pl.xlim(12,4)
    pl.yscale('log')
    pl.ylim(1,1e5)
    pl.legend(loc=0)
    pl.show()
