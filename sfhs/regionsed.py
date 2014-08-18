# Convert from sfhs of each region (for each metallicity)
#  to predicted fluxes.

import numpy as np
import bursty_sfh as bsp
from sedpy import attenuation, observate
from scipy.interpolate import interp1d

#maggies_to_cgs = 10**(-0.4*(2.406 + 5*np.log10([f.wave_effective for f in filterlist])))
lbins = np.arange(-20, 1, 0.025)
    
def regionsed(regions, sps, filters = ['galex_NUV'], lf_bases = None):
    """
    Given a cloud name (lmc | smc) and a filter, determine the
    broadband flux of each region in that cloud based on the SFHs of
    Harris & Zaritsky 2004 or 2009.  The sfhs are given separately for
    each metallicity; we treat each metallicity independently and sum
    the resulting spectra before determining the broadband flux.
    """
    header = regions.pop('header', None) #dump the header
    
    sps.params['sfh'] = 0 #ssp
    sps.params['imf_type'] = 0 #salpeter
    filterlist = observate.load_filters(filters)

    #set up output
    try:
        nb = len(lbins)
    except:
        nb = 1
    regname, alllocs = [], []
    allmags = np.zeros([len(regions), len(filterlist)])
    all_lfs = np.zeros([len(regions), nb])
            
    #loop over regions
    for j, (k, data) in enumerate(regions.iteritems()):
        spec, lf, wave = one_region(data['sfhs'], data['zmet'],
                                    sps, lf_bases = lf_bases)

        #project filters
        mags = observate.getSED(wave, spec * bsp.to_cgs, filterlist = filterlist)
        mags = np.atleast_1d(mags)
        allmags[j,:] = mags
        all_lfs[j,:] = lf
        regname.append(k)
        alllocs += [data['loc'].strip()]
        
    return alllocs, regname, allmags, all_lfs

def one_region(sfhs, zmet, sps, t_lookback = 0, lf_bases = None):
    """
    Get the spectrum and AGB LF of one region, given SFHs for each
    metallicity, a stellar population object, and lf_basis
    """
    spec = np.zeros(sps.wavelengths.shape[0])
    lf = 0
    mstar = 0
    
    nsfh = len(sfhs)
    if lf_bases is None:
        lf_bases = nsfh * [None]
        
    #loop over metallicities for each region
    for i, (sfh, lf_basis) in enumerate(zip(sfhs, lf_bases)):
        #choose nearest metallicity
        zindex = np.abs(sps.zlegend - zmet[i]).argmin() + 1
        sps.params['zmet'] = np.clip(zindex, 1, 5)
        # Put sfh in linear units, adjusting most recent time bin
        sfh['t1'] = 10.**sfh['t1']
        sfh['t2'] = 10.**sfh['t2']
        sfh['sfr'][0] *=  1 - (sfh['t1'][0]/sfh['t2'][0])
        sfh[0]['t1'] = 0.
        mtot = ((sfh['t2'] - sfh['t1']) * sfh['sfr']).sum()

        # Convert into a high resolution sfh,
        #  with *no* intrabin sfr variations (f_burst =0)
        lt, sfr, fb = bsp.burst_sfh(f_burst = 0., sfh = sfh,
                                    fwhm_burst = 0.05,  contrast = 1.,
                                    bin_res = 10.)

        # Get the intrinsic spectrum for this metallicity SFH
        #  and add to the total spectrum
        wave, zspec, aw = bsp.bursty_sps(t_lookback, lt, sfr, sps, av = None, dav = None)
        spec += zspec[0,:]
        mstar += mtot
        # Get the agb LF from this mettalicity interpolate onto
        # lbins, and add to total LF
        if lf_basis is not None:
            bins, zlf, aw = bsp.bursty_lf(t_lookback, lt, sfr, lf_basis)
            lf8 = interp1d(bins, zlf, bounds_error =False)
#            lf += lf8(lf_basis['bins'])[0,:]
            lf += lf8(lbins)[0,:]
    return spec, lf, wave
