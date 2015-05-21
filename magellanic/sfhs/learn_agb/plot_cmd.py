import numpy as np
import matplotlib.pyplot as pl

import fsps
from magellanic import cmdutils, lfutils, sputils


sps = fsps.StellarPopulation(compute_vega_mags=True)
sps.params['sfh'] = 0
sps.params['imf_type'] = 0
sps.params['tpagb_norm_type'] = 2 #VCJ
sps.params['add_agb_dust_model'] = True
sps.params['agb_dust'] = 1.0

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
    isoc_phase = cmdutils.agb_select_function(full_isoc)
    #select only C-AGB
    isoc_c = cmdutils.agb_select_function(full_isoc, composition=1.0)
    
    pcmd, ages = cmdutils.partial_cmds(isoc_dat, tuple(color), tuple(mag))
    lpcmds = combine_partial_cmds(pcmd, ages, esfh)
    pcmd_phase, ages_phase = cmdutils.partial_cmds(isoc_phase, tuple(color), tuple(mag))
    lpcmds_phase = combine_partial_cmds(pcmd_phase, ages_phase, esfh)
    pcmd_c, ages_c = cmdutils.partial_cmds(isoc_c, tuple(color), tuple(mag))
    lpcmds_c = combine_partial_cmds(pcmd_c, ages_c, esfh)

    mtot = mass.sum(axis=-1)
    
    fig, axes = pl.subplots(1, 3)
    im = axes[0].imshow(np.log10((lpcmds * mtot[:,None, None]).sum(axis=0).T),
                        interpolation='nearest',
                        extent=[color[-1].min(), color[-1].max(), mag[-1].max()+dm, mag[-1].min()+dm])
    im = axes[1].imshow(np.log10((lpcmds_phase * mtot[:,None, None]).sum(axis=0).T),
                        interpolation='nearest',
                        extent=[color[-1].min(), color[-1].max(), mag[-1].max()+dm, mag[-1].min()+dm])
    im = axes[2].imshow(np.log10((lpcmds_c * mtot[:,None, None]).sum(axis=0).T),
                        interpolation='nearest',
                        extent=[color[-1].min(), color[-1].max(), mag[-1].max()+dm, mag[-1].min()+dm])

    [ax.set_xlabel('J-K') for ax in axes]
    [ax.set_ylabel('K') for ax in axes]

    k0, k1, k2 = cmdutils.cioni_klines(color[-1], met, dm)
    [ax.plot(color[-1], k, label = 'K1') for ax in axes
     for k, l in zip([k0, k1, k2], ['K0',' K1', 'K2'])]
    [ax.set_ylim(mag[-1].max()+dm, mag[-1].min() + dm) for ax in axes]

    fig.show()

    nage, nc, nm = lpcmds.shape
    xticks = np.arange(0, nc+1, 5)
    yticks = np.arange(0, nm+1, 5)
    clabels = np.array(['{0:3.1f}'.format(l) for l in color[-1]])
    mlabels = np.array(['{0:3.1f}'.format(l+dm) for l in mag[-1]])
