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
        certain stellar types are desired in the CMD then isoc_dat
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
        The SFH given in terms of the mass formed in each age bin,
        ndarray.

    :param esfh:
        A structured array with the fields ``'t1'`` and ``'t2'``
        giving the beginning and end of each bin in the SFH (in
        lookback time, log(yrs) )

    :returns cmd:
        Array of total number of stars (weight*mtot) in bins of color
        and magnitude.
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
    nc, nm = pcmds.shape[-2:]
    lores_cmds = np.zeros([nsfh, nc, nm])
    if len(ages) == 0:
        return lores_cmds
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


def load_data(cloud='lmc'):
    """Read the Harris and Zaritsky SFH data.

    :param cloud:
        String, 'lmc' or 'smc'
    
    :returns mass:
        An array of masses formed in each (region, metallicity, age).
        ndarray of shape (nreg, nmet, nage), units of M_sun.

    :returns rnames:
        A list of region names.

    :returns mets:
        A list of metallicities used in the SFHs, linear units where
        solar=0.019.

    :returns example_sfhs:
        A structure giving the temporal binning scheme (i.e. the
        ages), with the beginning and end of each age bin given by the
        fileds ``'t1'`` and ``'t2'`` respectively, in units of
        log(years).
    """
    from magellanic import datautils

    #SFH data
    regions = mcutils.mc_regions(cloud)
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
    #cloud, dm = 'smc', 18.89
    cloud, dm = 'lmc', 18.49
    mass, _, mets, esfh = load_data(cloud)

    # Define a function for selecting certain "stars" from the
    # isochrone.  Only stars selected by this function will contribute
    # to the CMD
    def select_function(isochrones, cloud=None, **extras):
        """Select only certain stars from the isochrones.
        """
        c, o, b, x = cmdutils.boyer_cmd_classes(isochrones, cloud=cloud,
                                                **extras)
        # No filtering
        #return isochrones
        #Cstars only
        select = (b | x) & ~o 
        return isochrones[select]
    
    # Choose colors, mags as lists of lists.  The inner lists should
    # be passable as ``color`` and ``mag`` to the make_cmd() method
    # defined above.
    # Here we are doing j and ks but with the same j-k color in both
    # cases.
    colors = 2 * [['2mass_j', '2mass_ks', np.linspace(-1, 4.0, 501) ]]
        
    mags = [['2mass_ks', np.linspace(7, 14, 281) - dm],
            ['2mass_j', np.linspace(7, 14, 281) - dm]
            ]         

    # Build partial CMDs for each metallicity and project them onto
    # the SFH for that metallicity.  Outer loop should always be
    # metallicity to avoid excessive calls to sps.isochrones(), which
    # are expensive.
    cmd = []
    for i,z in enumerate(mets):
        cmd.append([])
        # Total (summed over regions) cloud SFH for this metallicity
        mtot = mass[:,i,:].sum(axis=0)
        
        # Isochrone data for this metallicity
        sps.params['zmet'] = np.argmin(np.abs(sps.zlegend - z)) + 1
        full_isoc = sps.isochrones()

        # Here you can loop over anything that doesn't affect the
        # generation of the isochrone data to make cmds
        for color, mag in zip(colors, mags):
            #filter the isochrone
            isoc_dat = select_function(full_isoc, cloud=cloud)
            cmd_zc = make_cmd(isoc_dat, tuple(color), tuple(mag), mtot, esfh)
            cmd[i].append(cmd_zc)

    # This makes an array of shape (nmet, nband, ncolor-1, nmag-1)
    # NB: this will not work if there are not the same number of bins
    # in each color or in each mag, though you can still use ``cmd``
    # in list form in that case.
    cmd = np.array(cmd)

    ##### OUTPUT #######
    
    # Plot output CMDs
    cfig, caxes = pl.subplots( 1, len(mags) )
    for j, (color, mag) in enumerate(zip(colors, mags)):
        ax = caxes.flat[j]
        im = ax.imshow(np.log10(cmd[:,j,...].sum(axis=0).T), interpolation='nearest',
                       extent=[color[-1].min(), color[-1].max(),
                               mag[-1].max()+dm, mag[-1].min()+dm])
        ax.set_xlabel('{0} - {1}'.format(color[0], color[1]))
        ax.set_ylabel('{0}'.format(mag[0]))
        ax.set_title(cloud.upper())
    cfig.show()

    # Plot output sAGB number
    #efig, eax = pl.subplots(


    ##### Observed CMD #####
    from magellanic.sfhs.datautils import cloud_cat, catalog_to_cmd
    from copy import deepcopy
    cat, cols = cloud_cat(cloud, convert=True)
    ofig, oaxes = pl.subplots( 1, len(mags) )
    for j, (color, mag) in enumerate(zip(colors, mags)):
        ax = oaxes.flat[j]
        appmag = deepcopy(mag)
        appmag[-1] += dm
        ocmd = catalog_to_cmd(cat, color, appmag)#, catcols=cols)
        im = ax.imshow(np.log10(ocmd.T), interpolation='nearest',
                       extent=[color[-1].min(), color[-1].max(),
                               appmag[-1].max(), appmag[-1].min()],
                        aspect='auto')
        ax.set_xlabel('{0} - {1}'.format(color[0], color[1]))
        ax.set_ylabel('{0}'.format(mag[0]))
        ax.set_title('Obs ' + cloud.upper())
    ofig.show()

    
    # Plot output CLFs
    # NB: the CLF does not include stars brighter than the brightest
    # bin in mag, or outside the color range
    lfig, laxes = pl.subplots()
    for j, mag in enumerate(mags):
        # marginalize over metallicity and color
        lf = cmd[:,j,:,:].sum(axis=0).sum(axis=-2)
        clf = np.cumsum(lf)
        x = mag[-1][1:]
        laxes.plot(x + dm, clf, label=mag[0])
    laxes.set_xlabel('m')
    laxes.set_ylabel('N(<m)')
    laxes.set_yscale('log')
    laxes.set_title(cloud.upper())
    laxes.legend(loc=0)
    lfig.show()

