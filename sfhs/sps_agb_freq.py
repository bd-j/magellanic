import numpy as np
import fsps, mcutils, sputils

def select_function(isoc_dat):
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
    print(select.sum())
    return isoc_dat[select]

def cmd_select_function(isoc_dat):
    """Trying to follow the Boyer et al. 2011 color cuts for AGB stars
    """
    trgb_lmc = {'k': 11.94,'i1':11.9, 'dm':18.4}
    #trgb_smc = {'k': 12.6, 'i1':12.6, 'dm':18.9}
    trgb = trgb_lmc
    
    j = isoc_dat['2mass_j'] + trgb['dm']
    k = isoc_dat['2mass_ks'] + trgb['dm']
    i1 = isoc_data['irac_1'] + trgb['dm']
    i4 = isoc_data['irac_4'] + trgb['dm']
    
    xagb = ((i1 < trgb['i1']) &
            (((j-i1) > 3.1) | ((i1-i4) > 0.8)) &
            (i4 < (12.0 - 0.43 * (j-i4))) &
            (i4 < (11.5 - 1.33 * (i1-i4)))
            )
        
    cioni = ((k < (-0.48 * (j-k) + 13)) &
              (k > (-13.33 * (j-k) + 24.666))
            )
    boyer = (cioni &
             ((k < trgb['k']) | (i1 < trgb['i1']))
            )
    cstars = (boyer & ~xagb &
              (k < (-13.333 * (j-k) + 28.4))
              )
    ostars = (boyer & ~xagb &
              (k > (-13.333 * (j-k) + 28.4))
              )

    select = boyer | xagb
    
    return isoc_dat[select]          
            


def sps_expected(isoc, esfh):
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


sps = fsps.StellarPopulation(compute_vega_mags=True)
sps.params['sfh'] = 0
sps.params['imf_type'] = 0
sps.params['tpagb_norm_type'] = 2 #VCJ
sps.params['add_agb_dust_model'] = True
sps.params['agb_dust'] = 1.0

def main(cloudname, select_function=select_function, **fsps_kwargs):

    for k, v in fsps_kwargs.iteritems():
        sps.params[k] = v
        
    cloud = cloudname.lower()
    if cloud == 'smc':
        regions = mcutils.smc_regions()
        zcloud = 0.004
    elif cloud == 'lmc':
        regions = mcutils.lmc_regions()
        zcloud = 0.008
        #zcloud = 0.5 * 0.019
        
    esfh = regions['AA']['sfhs'][0]
    sps.params['zmet'] = np.abs(zcloud - sps.zlegend).argmin() + 1
    zactual = sps.zlegend[sps.params['zmet'] - 1]
    print(r'Using $Z={0}Z_\odot$'.format(zactual/0.019))
    isoc = sps.isochrones()
    agbisoc = select_function(isoc)
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

    
