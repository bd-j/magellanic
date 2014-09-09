# Convert from sfhs of each region (for each metallicity)
#  to predicted fluxes.

import numpy as np
import sputils as bsp
from scipy.interpolate import interp1d

lfbins = np.arange(-20, 1, 0.025)
lsun, pc = 3.846e33, 3.085677581467192e18
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2 )

def one_region_sed(sfhs, zmet, sps, t_lookback = 0):
    """
    Get the spectrum of one region, given SFHs for each
    metallicity, a stellar population object, and lf_basis
    """
    spec = np.zeros(sps.wavelengths.shape[0])
    mstar = 0
            
    #loop over metallicities for each region
    for i, sfh in enumerate(sfhs):
        #choose nearest metallicity
        zindex = np.abs(sps.zlegend - zmet[i]).argmin() + 1
        sps.params['zmet'] = np.clip(zindex, 1, len(sps.zlegend))
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
        wave, zspec, aw, zmass = bsp.bursty_sps(t_lookback, lt, sfr, sps)
        spec += zspec[0,:]
        mstar += zmass[0]
    return spec, wave, mstar


def one_region_lfs(sfhs, lf_bases, t_lookback = 0):
    """
    Get the AGB LF of one region, given a list of SFHs for each
    metallicity and a list of lf_bases for each metallicity.
    """
    lf, total_lf, logages = [], 0., []
    #loop over metallicities for each region
    for i, (sfh, lf_basis) in enumerate(zip(sfhs, lf_bases)):
        # Get the weighted agb LFs from this metallicity SFH, sum over
        # ages, interpolate onto lfbins, and add to total LF
        bins, wlf = one_sfh_lfs(sfh, lf_basis, t_lookback=t_lookback)
        wlf = wlf[0,:,:] #restrict to one lookback time
        lf8 = interp1d(bins, wlf, bounds_error=False, fill_value=0.0)
        dat = lf8(lfbins)
        total_lf += dat.sum(axis=0)
        lf += [dat]
        logages += [lf_basis['ssp_ages']]
    return lf, total_lf, logages

def one_sfh_lfs(sfh, lf_basis, t_lookback = 0):
    """
    Get the AGB LFs for one SFH.

    :returns bins: ndarray shape (nage, nmagbin)
        The LFs, weighted by the SFH, for each age

    :returns weighted_lfs: ndarray shape (nt, nage, nmagbin) The LFs,
        weighted by the SFH, for each age (nage) and requested
        lookback time (nt).
    """
    bins, lf, ssp_ages = lf_basis['bins'], lf_basis['lf'], 10**lf_basis['ssp_ages']
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
                                bin_res = 20.)

    # Get the weights for each age and muliply by the LFs
    aw = bsp.sfh_weights(lt, sfr, ssp_ages, np.atleast_1d(t_lookback))
    weighted_lfs = (lf[None,:,:] * aw[:,:,None])
        
    return bins, weighted_lfs

def all_region_sed(regions, sps, filters = ['galex_NUV'], lf_bases = None):
    """
    Given a cloud name (lmc | smc) and a filter, determine the
    broadband flux of each region in that cloud based on the SFHs of
    Harris & Zaritsky 2004 or 2009.  The sfhs are given separately for
    each metallicity; we treat each metallicity independently and sum
    the resulting spectra before determining the broadband flux.
    """
    
    from sedpy import observate
    header = regions.pop('header', None) #dump the header
    
    sps.params['sfh'] = 0 #ssp
    sps.params['imf_type'] = 0 #salpeter
    filterlist = observate.load_filters(filters)

    #set up output
    try:
        nb = len(lfbins)
    except:
        nb = 1
    regname, alllocs = [], []
    allmags = np.zeros([len(regions), len(filterlist)])
    all_lfs = np.zeros([len(regions), nb])
            
    #loop over regions
    for j, (k, data) in enumerate(regions.iteritems()):
        spec, lf, wave = one_region_sed(data['sfhs'], data['zmet'],
                                        sps, lf_bases = lf_bases)

        #project filters
        mags = observate.getSED(wave, spec * bsp.to_cgs, filterlist = filterlist)
        mags = np.atleast_1d(mags)
        allmags[j,:] = mags
        all_lfs[j,:] = lf
        regname.append(k)
        alllocs += [data['loc'].strip()]
        
    return alllocs, regname, allmags, all_lfs
