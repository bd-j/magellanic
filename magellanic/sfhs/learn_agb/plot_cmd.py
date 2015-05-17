import numpy as np
import matplotlib.pyplot as pl

import fsps
from magellanic import sps_freq, lfutils, sputils

sps = fsps.StellarPopulation(compute_vega_mags=True)
sps.params['sfh'] = 0
sps.params['imf_type'] = 0
sps.params['tpagb_norm_type'] = 2 #VCJ
sps.params['add_agb_dust_model'] = True
sps.params['agb_dust'] = 1.0


def partial_cmds(isoc, color, mag):
    agecol = 'age'
    ages = np.unique(isoc[agecol])
    cmds = []
    for age in ages:
        thisage = isoc[agecol] == age
        cmds.append(lfutils.isocdata_to_cmd(isoc[thisage], color, mag))
    cmds = np.array(cmds)
    oo = np.sort(ages)
    
    return cmds, ages

def combine_partial_cmds(cmds, ages, sfh):
    asfh = sfh.copy()
    asfh['t1'] = 10**asfh['t1']
    asfh['t2'] = 10**asfh['t2']
    for i in range(len(asfh)):
        if asfh['t2'][i] < 10**ages.min():
            #print(i)
            continue
        asfh['sfr'] = 0
        asfh['sfr'][i] = 1.0/(asfh['t2'][i] - asfh['t1'][i])
        lt, sfr, fact = sputils.burst_sfh(f_burst=0.0, sfh = asfh, bin_res=20)
        aw = sputils.sfh_weights(lt, sfr, 10**ages)
        total_cmds[i] = (aw[0,:] * cmds).sum()


if __name__ == "__main__":
    dm, z = 18.9, 0.008
    color = ['2mass_j', '2mass_ks', np.arange(-1, 3.0, 0.1)]
    mag = ['2mass_ks', np.arange(7, 12, 0.1) - dm]

    sps.params['zmet'] = np.argmin(np.abs(sps.zlegend - z)) + 1
    
    full_isoc = sps.isochrones()
    #isoc_dat = select_function(full_isoc)
    isoc_dat = full_isoc
    pcmd, ages = partial_cmds(isoc_dat, tuple(color), tuple(mag))
    
