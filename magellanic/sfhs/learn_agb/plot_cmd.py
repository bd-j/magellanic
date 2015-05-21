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
    oo = np.argsort(ages)
    
    return cmds[oo, :,:], ages[oo]

def combine_partial_cmds(pcmds, ages, sfh):
    """
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

def plot_cioni(ax, dm, delta_jk=0):
    j_m_k = np.arange(-1, 3, 0.1)
    cioni = (-0.48 * (j_m_k + delta_jk) + 13)
    cioni2 = (-13.33 * (j_m_k + delta_jk) + 24.666)
    ax.plot()

if __name__ == "__main__":
    
    from learn_agb import load_data
    cloud='smc'
    if cloud =='lmc':
        dm, z, met = 18.49, 0.008, -0.7
    elif cloud =='smc':
        dm, z, met = 18.89, 0.004, -0.3
        
    _, mass, nex, esfh = load_data(cloud)
    
    color = ['2mass_j', '2mass_ks', np.arange(-1, 4.0, 0.01)]
    mag = ['2mass_ks', np.arange(7, 14, 0.025) - dm]

    sps.params['zmet'] = np.argmin(np.abs(sps.zlegend - z)) + 1
    full_isoc = sps.isochrones()
    #isoc_dat = select_function(full_isoc)
    isoc_dat = full_isoc
    isoc_phase = sps_freq.agb_select_function(full_isoc)

    pcmd, ages = partial_cmds(isoc_dat, tuple(color), tuple(mag))
    lpcmds = combine_partial_cmds(pcmd, ages, esfh)
    pcmd_phase, ages_phase = partial_cmds(isoc_phase, tuple(color), tuple(mag))
    lpcmds_phase = combine_partial_cmds(pcmd_phase, ages_phase, esfh)

    mtot = mass.sum(axis=-1)
    cmd_tot = lpcmds * mtot[:,None,None]
    
    
    nage, nc, nm = lpcmds.shape
    xticks = np.arange(0, nc+1, 5)
    yticks = np.arange(0, nm+1, 5)
    clabels = np.array(['{0:3.1f}'.format(l) for l in color[-1]])
    mlabels = np.array(['{0:3.1f}'.format(l+dm) for l in mag[-1]])
    
    
    fig, axes = pl.subplots(1, 2)
    im = axes[0].imshow(np.log10(cmd_tot[0:,:,:].sum(axis=0).T),
                        interpolation='nearest',
                        extent=[color[-1].min(), color[-1].max(), mag[-1].max()+dm, mag[-1].min()+dm])
    im = axes.imshow(np.log10((lpcmds_phase * mtot[:,None, None]).sum(axis=0).T),
                        interpolation='nearest',
                        extent=[color[-1].min(), color[-1].max(), mag[-1].max()+dm, mag[-1].min()+dm])

    [ax.set_xlabel('J-K') for ax in axes]
    [ax.set_ylabel('K') for ax in axes]

    k0, k1, k2 = sps_freq.cioni_klines(color[-1], met, dm)
    [ax.plot(color[-1], k, label = 'K1') for ax in axes
     for k, l in zip([k0, k1, k2], ['K0',' K1', 'K2'])]
    [ax.set_ylim(mag[-1].max()+dm, mag[-1].min() + dm) for ax in axes]

    fig.show()
