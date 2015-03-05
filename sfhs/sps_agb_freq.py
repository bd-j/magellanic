import numpy as np
import fsps, mcutils

def select_function(isoc_dat):
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
    print(select.sum())
    return isoc_dat[select]


def sps_expected(isoc, esfh):
    """
    :param isoc:
        An isochrone as a numpyt structured array, as returned by
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

if __name__ == "__main__":

    cloud = 'lmc'
    regions = mcutils.lmc_regions()
    esfh = regions['AA']['sfhs'][0]
    zcloud = 0.019 * 0.4
    sps.params['zmet'] = np.abs(zcloud - sps.zlegend).argmin() + 1
    isoc = sps.isochrones()
    agbisoc = select_function(isoc)
    ssp_ages, ssp_nexpected = sps_expected(agbisoc, esfh)
    
    dt = np.concatenate([[10**ssp_ages[0]], 10**ssp_ages[1:] - 10**ssp_ages[:-1]])
    nexpected = np.zeros(len(esfh))
    
    for i, (start, stop) in enumerate(zip(esfh['t1'], esfh['t2'])):
        this = (ssp_ages <= stop) & (ssp_ages > start)
        if this.sum() == 0:
            continue
        wght = dt.copy()
        #adjust end weights
        mi, ma = ssp_ages[this].argmin(), ssp_ages[this].argmax()
        wght[this][mi] = 10**ssp_ages[this][mi] - 10**start
        wght[this][ma] += 10**stop - 10**ssp_ages[this][ma]
        #need to do a weighted sum, with weights given by dt
        nexpected[i] = (ssp_nexpected[this] * wght[this]).sum()/wght[this].sum()
        print(10**stop - 10**start, wght[this].sum())

    
