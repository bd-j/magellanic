import numpy as np
import matplotlib.pyplot as pl

import fsps
from magellanic import cmdutils, sputils, mcutils

def make_cmd(isoc_dat, color, mag, mtot, esfh):
    """Make a total CMD by producing partial CMDs and then projecting
    them onto the SFH.

    :param isoc_dat:
        Isochrone data as a structured array.  This is usually the
        result of fsps.StellarPopulation().isochrones().  If only
        certain stellar types are desired int he CMD then isoc_dat
        should be pre-filtered.

    :param color:
        A tuple giving the bandnames and bin edges for the color.  It
        should have the form ``('band1', 'band2', bins)`` where
        ``bins`` is ndarray of bin edges and ``'band1'`` and
        ``'band2'`` are the names of the FSPS filters that form color
        'band1-band2'.
        
    :param mag:
        A tuple of absolute magnitude bins of the form ``('band',bins)``
        where bins is an ndarray of bin edges and `band' is the filter.

    :param mtot:
        The SFH given in terms of the mass formed in each bin, ndarray.

    :param esfh:
        A structured array with the fields ``'t1'`` and ``'t2'``
        giving the beginning and end of each bin in the SFH (in
        lookback time, log(yrs) )

    :returns cmd:
        Array of total weights (N_stars) in bins of color and magnitude.
    """
    pcmd, ages = cmdutils.partial_cmds(isoc_dat, tuple(color), tuple(mag))
    lpcmds = rebin_partial_cmds(pcmd, ages, esfh)
    return (lpcmds * mtot[:,None, None]).sum(axis=0)

def rebin_partial_cmds(pcmds, ages, sfh):
    """Combine the SSp partial cmds into broader bins given by esfh
    """
    asfh = sfh.copy()
    asfh['t1'] = 10**asfh['t1']
    asfh['t2'] = 10**asfh['t2']
    nsfh = len(asfh)
    nage, nc, nm = pcmds.shape
    lores_cmds = np.zeros([nsfh, nc, nm])
    for i in range(len(asfh)):
        if asfh['t2'][i] < 10**ages.min():
            #print(i)
            continue
        asfh['sfr'] = 0
        asfh['sfr'][i] = 1.0/(asfh['t2'][i] - asfh['t1'][i])
        lt, sfr, fact = sputils.burst_sfh(f_burst=0.0, sfh = asfh, bin_res=20)
        aw = sputils.sfh_weights(lt, sfr, 10**ages)
        lores_cmds[i,:,:] = (aw[0,:][:,None,None] * pcmds).sum(axis = 0)
        
    return lores_cmds


def load_data(cloud):
    """Read the Harris and Zaritsky data.

    :param cloud:
        'lmc' or 'smc'
    
    :returns mass:
        An array of masses formed in each (region, metallicity, age).
        ndarray of shape (nreg, nmet, nage)

    :returns rnames:
        A list of region names

    :returns mets:
        a list of metallicities used in the SFHs,

    :returns example_sfhs:
        and a structure giving the temporal binning scheme (the ages)
    """
    from magellanic import datautils
    import sedpy.ds9region as ds9

    #SFH data
    if cloud.lower() == 'lmc':
        print('doing lmc')
        regions = mcutils.lmc_regions()
    else:
        print('doing smc')
        regions = mcutils.smc_regions()
    regions.pop('header')

    #convert SFHs to mass formed per bin
    mass, rnames = [], []
    for name, dat in regions.iteritems():
        mass += [[ s['sfr'] * (10**s['t2'] - 10**s['t1'])  for s in dat['sfhs'] ]]
        rnames.append(name)
    #get metallicity and temporal info for the SFHs
    mets = regions[rnames[0]]['zmet']
    example_sfh = regions[rnames[0]]['sfhs'][0]
    
    return np.array(mass), rnames, mets, example_sfh 
    
if __name__ == "__main__":

    #SPS
    sps = fsps.StellarPopulation(compute_vega_mags=True)
    sps.params['sfh'] = 0
    sps.params['imf_type'] = 0
    sps.params['tpagb_norm_type'] = 2 #VCJ
    sps.params['add_agb_dust_model'] = True
    sps.params['agb_dust'] = 1.0

    # Cloud SFHs
    cloud, dm = 'smc', 18.89
    mass, _, mets, esfh = load_data(cloud)
    mtot = mass.sum(axis=-1)

    # Choose colors, mags and define isochrone filtering
    colors = 2 * [['2mass_j', '2mass_ks', np.arange(-1, 4.0, 0.01) ]]
              
    mags = [['2mass_ks', np.arange(7, 14, 0.025) - dm],
            ['2mass_j', np.arange(7, 14, 0.025) - dm]
            ]
            
    def select_function(isochrones, cloud=None):
        """Select only certain stars from the isochrones.
        """
        # No filtering
        return isochrones

    # Build partial CMDs for each metallicity and project them onto
    # the SFH for that metallicity.  Outer loop should always be
    # metallicity to avoid excessive calls to sps.isochrones(), which
    # are expensive.
    cmd = len(mets) * [[]]
    for i,z in enumerate(mets):
        # Total cloud SFH for this metallicity
        mtot = mass[:,i,:].sum(axis=0)
        # Isochrone data for this metallicity
        sps.params['zmet'] = np.argmin(np.abs(sps.zlegend - z)) + 1
        full_isoc = sps.isochrones()
        isoc_dat = select_function(full_isoc)
        # Here you can loop over anything that doesn't affect the generation of the isochrone data
        for color, mag in zip(colors, mags):
            cmd_zc = make_cmd(isoc_dat, tuple(color), tuple(mag), mtot, esfh)
            cmd[i].append(cmd_zc)
        
    
    
 
