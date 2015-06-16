import numpy as np
import mcutils, sputils


def isocdata_to_cmd(isoc_dat, color, mag):
    """Make a CMD from isochrone data.

    :param isoc_data:
        The isochrone data, as returned by
        StellarPopulation.isochrones().  It should be prefiltered by
        age and perhaps stellar type (phase).

    :param color:
        A tuple giving the bandnames and bin edges for the color.  It
        should have the form ``('band1', 'band2', bins)`` where
        ``bins`` is ndarray of bin edges and ``'band1'`` and
        ``'band2'`` are the names of the FSPS filters that form color
        'band1-band2'.
        
    :param mag:
        A tuple of absolute magnitude bins of the form ``('band',bins)``
        where bins is an ndarray of bin edges and `band' is the filter.

    :returns cmd:
        A 2-d numpy array of shape (nc, nm) giving the color magnitude
        diagram
    """
    c = isoc_dat[color[0]] - isoc_dat[color[1]]
    m = isoc_dat[mag[0]]
    cmd, _, _ = np.histogram2d(c, m, bins=[color[2], mag[1]],
                         weights=10**isoc_dat['log(weight)'])
    return cmd

def partial_cmds(isoc, color, mag):
    """Make a partial CMDs (i.e. a series of CMDs of SSPSs) from
    isochrone data.

    :param isoc_data:
        The isochrone data, as returned by
        StellarPopulation.isochrones().  It should be prefiltered by
        stellar type (phase) if you want only the cmds for particular
        stellar types.

    :param color:
        A tuple giving the bandnames and bin edges for the color.  It
        should have the form ``('band1', 'band2', bins)`` where
        ``bins`` is ndarray of bin edges and ``'band1'`` and
        ``'band2'`` are the names of the FSPS filters that form color
        'band1-band2'.
        
    :param mag:
        A tuple of absolute magnitude bins of the form ``('band',bins)``
        where bins is an ndarray of bin edges and `band' is the filter.

    :returns cmds:
        A 3-d numpy array of shape (nage, nc, nm) giving the binned
        color magnitude diagrams for each age.
    """
    agecol = 'age'
    ages = np.unique(isoc[agecol])
    cmds = []
    for age in ages:
        thisage = isoc[agecol] == age
        cmds.append(isocdata_to_cmd(isoc[thisage], color, mag))
    cmds = np.array(cmds)
    oo = np.argsort(ages)
    
    return cmds[oo, :,:], ages[oo]

def rebin(a, shape):
    """Rebin array to new shape.  New shape must be integer fractions
    of the old shape
    """
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def sps_expected(isoc):
    """
    :param isoc:
        An isochrone as a numpy structured array, as returned by
        fsps.StellarPopulation.isochrones().  It must have a
        ``log(weight)`` field.
        
    :returns ssp_ages:
        The ages of the SSPs, log of yrs

    :returns ssp_nexpected:
        The sum of the imf_weights for isochrone points with this age.
    """
    logages = isoc['age']
    ssp_ages = np.unique(logages)
    total_weights = [(10**(isoc[logages == thisage]['log(weight)'])).sum() for
                     thisage in ssp_ages]
    return ssp_ages, np.array(total_weights)

def agb_select_function(isoc_dat, composition=0, **extras):
    """
    Here's a function that selects certain rows from the full
    isochrone data and returns them.  The selection criteria can
    involve any of the columns given by the isochrone data, including
    magnitudes (or colors) as well as things like logg, logL, etc.
    """
    #select only objects cooler than 4000K and in tp-agb phase
    select = ( #(isoc_dat['logt'] < np.log10(4000.0)) &
               (isoc_dat['phase'] == 5.0)
               )
    if composition < 0:
        select = select & (isoc_dat['composition'] < -composition)
    elif composition > 0:
        select = select & (isoc_dat['composition'] > composition)
    print(select.sum())
    return isoc_dat[select]


def agb_select_function_villaume(isoc_dat, **extras):
    """
    Here's a function that selects certain rows from the full
    isochrone data and returns them.  The selection criteria can
    involve any of the columns given by the isochrone data, including
    magnitudes (or colors) as well as things like logg, logL, etc.
    """
    #select only objects cooler than 4000K and in tp-agb phase
    select = ( (isoc_dat['logt'] < np.log10(4000.0)) &
               (isoc_dat['phase'] != 6.0) &
               (isoc_dat['age'] > 6)
               )
    
    print(select.sum())
    return isoc_dat[select]

def agb_select_function_cmd(isoc_dat, **kwargs):
    
    c, o, boyer, xagb = boyer_cmd_classes(isoc_dat, **kwargs)
    return isoc_dat[boyer | xagb]
    
def boyer_cmd_classes(isoc_dat, cloud='lmc', is_data_cat=False, **extras):
    """Boyer cmd cuts.
    """
    if cloud.lower() == 'lmc':
        cdat = {'trgb_k': 11.94, 'trgb_i1':11.9, 'dm':18.49, 'met':-0.3}
        delta_dm = 0.4
    elif cloud.lower() =='smc':
        cdat = {'trgb_k': 12.7, 'trgb_i1':12.6, 'dm':18.89, 'met':-0.7}
        delta_dm = 0.0


    j = isoc_dat['2mass_j'] + cdat['dm'] * int(is_data_cat)
    k = isoc_dat['2mass_ks'] + cdat['dm'] * int(is_data_cat)
    i1 = isoc_dat['irac_1'] + cdat['dm'] * int(is_data_cat)
    i4 = isoc_dat['irac_4'] + cdat['dm'] * int(is_data_cat)
    
    # Cioni
    k0, k1, k2 = cioni_klines(j-k, **cdat)
    cstar = (k < k0) &  (k > k2)
    ostar = (k < k0) &  (k > k1) & ~cstar
    cioni = ostar | cstar
    
    # Boyer trgb cut
    boyer = (cioni &
             ((k < cdat['trgb_k']) | (i1 < cdat['trgb_i1']))
            )
    # Boyer xagb cut
    xagb = ((i1 < cdat['trgb_i1']) &
            (((j-i1) > 3.1) | ((i1-i4) > 0.8)) &
            ((i4 + delta_dm) < (12.0 - 0.43 * (j-i4))) &
            ((i4 + delta_dm) < (11.5 - 1.33 * (i1-i4)))
            )
    return cstar, ostar, boyer, xagb
        
def cioni_klines(color, met, dm, **extras):
    """The cioni classification criteria.  `color` should be j-ks
    """
    # We add the differential distance modulus to K0 but not to K1 or K2
    #
    k0 = -0.48 * color + 13.022 + 0.056 * met + (dm - 18.49)
    k1 = -13.333 * color + 25.293 + 1.568 * met 
    k2 = -13.333 * color + 29.026 + 1.568 * met
    return k0, k1, k2

def agb_select_function_cmd_old(isoc_dat, cloud='lmc', **extras):
    """Select AGBs using CMD cuts.
    """
    if cloud.lower() == 'lmc':
        return agb_select_function_cmd_lmc(isoc_dat, **extras)
    elif cloud.lower() == 'smc':
        return agb_select_function_cmd_smc(isoc_dat, **extras)
    else:
        raise ValueError('Invalid cloud designation')

def agb_select_function_cmd_lmc(isoc_dat, **extras):
    """Trying to follow the Boyer et al. 2011 color cuts for AGB stars
    """
    trgb = {'k': 11.94,'i1':11.9, 'dm':18.49}
    #difference between the SMC and LMC distance moduli
    delta_dm = 0.40
    
    j = isoc_dat['2mass_j'] + trgb['dm']
    k = isoc_dat['2mass_ks'] + trgb['dm']
    i1 = isoc_dat['irac_1'] + trgb['dm']
    i4 = isoc_dat['irac_4'] + trgb['dm']

    # Boyer X-AGB, accounting for distance differences
    xagb = ((i1 < trgb['i1']) &
            (((j-i1) > 3.1) | ((i1-i4) > 0.8)) &
            ((i4 + delta_dm) < (12.0 - 0.43 * (j-i4))) &
            ((i4 + delta_dm) < (11.5 - 1.33 * (i1-i4)))
            )
        
    # Cioni 2006a (LMC) cuts
    cioni = ((k < (-0.48 * (j-k) + 13)) &
             (k > (-13.33 * (j-k) + 24.666))
            )
    # Boyer trgb cut
    boyer = (cioni &
             ((k < trgb['k']) | (i1 < trgb['i1']))
            )
        
    # Cioni 2006a (LMC) K1 line, exluding xagbs
    cstars = (boyer & ~xagb &
              (k < (-13.333 * (j-k) + 28.4))
              )
    ostars = (boyer & ~xagb &
              (k > (-13.333 * (j-k) + 28.4))
              )

    select = boyer | xagb
    
    return isoc_dat[select]          
            
def agb_select_function_cmd_smc(isoc_dat, **extras):
    """Trying to follow the Boyer et al. 2011 color cuts for AGB stars
    For the SMC
    """
    trgb = {'k': 12.7, 'i1':12.6, 'dm':18.89}
    #difference between the SMC and LMC distance moduli
    delta_dm = -0.40
    #metallicity effect on j-k color
    # 0.056 * Z  (Z_LMC = -0.3, Z_SMC = -0.7)
    delta_jk = -0.05
    
    j = isoc_dat['2mass_j'] + trgb['dm']
    k = isoc_dat['2mass_ks'] + trgb['dm']
    i1 = isoc_dat['irac_1'] + trgb['dm']
    i4 = isoc_dat['irac_4'] + trgb['dm']
    
    xagb = ((i1 < trgb['i1']) &
            (((j-i1) > 3.1) | ((i1-i4) > 0.8)) &
            (i4 < (12.0 - 0.43 * (j-i4))) &
            ((i4 < (11.5 - 1.33 * (i1-i4))) | (((i1-i4) > 3) & (i4 < 7.51)))
            )
    #
        
    # Cioni 2006a (LMC) cuts adjusted for distance and metallicity
    
    cioni = (((k + delta_dm) < (-0.48 * (j - k + delta_jk) + 13)) &
              ((k + delta_dm) > (-13.33 * (j - k + delta_jk) + 24.666))
            )
    #these cuts already adjusted for distance through trgb differences
    boyer = (cioni &
             ((k < trgb['k']) | (i1 < trgb['i1']))
            )
    # Cioni 2006a (LMC) K1 line adjusted for distance and metallicity
    cstars = (boyer & ~xagb &
              ((k+ delta_dm) < (-13.333 * (j - k + delta_jk) + 28.4))
              )
    ostars = (boyer & ~xagb &
              ((k+ delta_dm) > (-13.333 * (j - k +delta_jk) + 28.4))
              )

    select = boyer | xagb
    
    return isoc_dat[select]          


def make_freq_prediction(cloudname, esfh, sps=None,
                         select_function=agb_select_function,
                         **kwargs):

    if sps is None:
        import fsps
        sps = fsps.StellarPopulation(compute_vega_mags=True)
        sps.params['sfh'] = 0
        sps.params['imf_type'] = 0
        sps.params['tpagb_norm_type'] = 2 #VCJ
        sps.params['add_agb_dust_model'] = True
        sps.params['agb_dust'] = 1.0

    for k, v in kwargs.iteritems():
        try:
            sps.params[k] = v
        except:
            pass
        
    cloud = cloudname.lower()
    if cloud == 'smc':
        #regions = mcutils.smc_regions()
        zcloud = 0.004
    elif cloud == 'lmc':
        #regions = mcutils.lmc_regions()
        zcloud = 0.008
        #zcloud = 0.5 * 0.019
        
    #esfh = regions['AA']['sfhs'][0]
    sps.params['zmet'] = np.abs(zcloud - sps.zlegend).argmin() + 1
    zactual = sps.zlegend[sps.params['zmet'] - 1]
    print(r'Using $Z={0}Z_\odot$'.format(zactual/0.019))
    isoc = sps.isochrones()
    agbisoc = select_function(isoc, cloud=cloud, **kwargs)
    ssp_ages, ssp_nexpected = sps_expected(agbisoc, esfh)
    
    dt = np.concatenate([[10**ssp_ages[0]], 10**ssp_ages[1:] - 10**ssp_ages[:-1]])
    nexpected = np.zeros(len(esfh))

    asfh = esfh.copy()
    asfh['t1'] = 10**asfh['t1']
    asfh['t2'] = 10**asfh['t2']
    for i in range(len(asfh)):
        if asfh['t2'][i] < 10**ssp_ages.min():
            #print(i)
            continue
        asfh['sfr'] = 0
        asfh['sfr'][i] = 1.0/(asfh['t2'][i] - asfh['t1'][i])
        lt, sfr, fact = sputils.burst_sfh(f_burst=0.0, sfh = asfh, bin_res=20)
        aw = sputils.sfh_weights(lt, sfr, 10**ssp_ages)
        nexpected[i] = (aw[0,:] * ssp_nexpected).sum()
        
#    for i, (start, stop) in enumerate(zip(esfh['t1'], esfh['t2'])):
#        this = (ssp_ages <= stop) & (ssp_ages > start)
#        if this.sum() == 0:
#            continue
#        wght = dt.copy()
#        #adjust end weights
#        mi, ma = ssp_ages[this].argmin(), ssp_ages[this].argmax()
#        wght[this][mi] = 10**ssp_ages[this][mi] - 10**start
#        wght[this][ma] += 10**stop - 10**ssp_ages[this][ma]
        #need to do a weighted sum, with weights given by dt
#        nexpected[i] = (ssp_nexpected[this] * wght[this]).sum()/wght[this].sum()
    return nexpected, zactual

    

if __name__ == "__main__":
    import fsps
    sps = fsps.StellarPopulation(compute_vega_mags=True)
    sps.params['sfh'] = 0
    sps.params['imf_type'] = 0
    sps.params['tpagb_norm_type'] = 2 #VCJ
    sps.params['add_agb_dust_model'] = True
    sps.params['agb_dust'] = 1.0


    cloud = []
    agb_norm_type = []
    selfn = []
