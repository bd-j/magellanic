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
        spec, wave, mass = rsed.one_region_sed(copy.deepcopy(sfhlist), total_zmet, sps)
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
            "Zmet={1}".format(sps.zlegend[sps.params['zmet']-1],zmet))

        ldat = isochrone_to_clfs(copy.deepcopy(isoc), lfbandnames,
                                 select_function=select_function)
        all_lf_base += [ldat]
    #use the SSP CLFs to generate a total LF (for each band)
    for i, band in enumerate(lfbandnames):
        lf_oneband = {}
        lf_base = [zbase[i] for zbase in all_lf_base]
        lfs_zt, lf, logages = rsed.one_region_lfs(copy.deepcopy(sfhlist), lf_base)
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

def make_clfs(cloud, tpagb_norm_type=2, select_function=None,
              sedfilters = None, **sps_kwargs):
    
    # These are the filters for integrated magnitudes of the object
    #sedfilters = ['galex_NUV', 'spitzer_irac_ch2',
    #              'spitzer_irac_ch4', 'spitzer_mips_24']
    
    # These are the filters for which you want LFs
    lffilters = ['2mass_ks','irac_2','irac_4']

    ########
    # Get the SFH
    ########
    import mcutils
    if cloud.lower() == 'lmc':
        print('doing lmc')
        dm = 18.5
        regions = mcutils.lmc_regions()
    elif cloud.lower() == 'smc':
        print('doing smc')
        dm = 18.9
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
    sps = fsps.StellarPopulation(compute_vega_mags=True)

    sps.params['add_agb_dust_model'] = True
    sps.params['tpagb_norm_type'] = tpagb_norm_type
    sps.params['sfh'] = 0
    sps.params['imf_type'] = 0.0 #salpeter
    sps.params['agb_dust'] = 1.0
    for k, v in sps_kwargs.iteritems():
        try:
            sps.params[k] = v
        except(KeyError):
            pass
    
    # Go from SFH to LFs and SED
    sed, clfs = zsfh_to_obs(total_sfhs, total_zmet,
                            lfbandnames=lffilters,
                            bandnames=sedfilters, sps=sps,
                            select_function=select_function)

    return clfs

if __name__ == '__main__':
    
    from sps_agb_freq import select_function as phase_select
    from sps_agb_freq import select_function_villaume as phase_select_v
    from sps_agb_freq import cmd_select_function_lmc, cmd_select_function_smc
    import datautils
    
    tptypes = [0,1,2]
    tpnames = ['MG08', 'CG10', 'VCJ14']
    clouds = ['lmc', 'smc']
    sel_fn = [phase_select_v, phase_select_v]
    title = 'Phase!=6, T<4000 Cut'
    #sel_fn = [cmd_select_function_lmc, cmd_select_function_smc]
    #title = 'Obs CMD Cuts'
    dm = [18.5, 18.9]
    
    fig, axes = pl.subplots(3, 2, figsize=(10,8))
    for j, cloud in enumerate(clouds):
        for ttype in tptypes:
            clfs = make_clfs(cloud, tpagb_norm_type=ttype,
                             select_function=sel_fn[j])
            for i, lf in enumerate(clfs):
                label = tpnames[ttype]
                axes[i,j].plot(lf['clf_mags']+dm[j], lf['clf'], label=label)
                axes[i,j].set_xlabel(lf['bandname'], )
                axes[i,j].set_xlim(12,4)
                axes[i,j].set_yscale('log')
                axes[i,j].set_ylim(0.1,1e6)
                
        axes[0,j].set_title(cloud.upper())
    #pl.legend(loc=0)
    axes[0,0].legend(loc=0, prop={'size':8})
    fig.suptitle(title)
    fig.show()
