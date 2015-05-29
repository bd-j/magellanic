import numpy as np
import matplotlib.pyplot as pl

import fsps
from magellanic import cmdutils, lfutils, sputils, mcutils

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
    from learn_agb import load_data, LinearModel, run_hmc
    cloud='lmc'
    if cloud =='lmc':
        dm, z, met = 18.49, 0.008, -0.7
    elif cloud =='smc':
        dm, z, met = 18.89, 0.004, -0.3
        
    rname, mass, nclass, esfh = load_data(cloud)
    mtot = mass.sum(axis=-1)

    # Get the isochrones
    sps.params['zmet'] = np.argmin(np.abs(sps.zlegend - z)) + 1
    full_isoc = sps.isochrones()

    # Select the stars of interest from the isochrone
    isoc_cx = cmdutils.agb_select_function(full_isoc)#, composition=1.0)

    # Build the partial CMDs
    color = ['2mass_j', '2mass_ks', np.arange(-1, 4.0, 0.01)]
    mag = ['2mass_ks', np.arange(7, 14, 0.025) - dm]
    
    pcmd, ages = cmdutils.partial_cmds(isoc_cx, tuple(color), tuple(mag))
    pcmd_lores = combine_partial_cmds(pcmd, ages, esfh)

    sN = pcmd_lores.sum(axis=-1).sum(axis=-1)
    npred = np.dot(sN, mass)
    nsamp = np.random.poisson(npred, len(npred))
    
    # Solve the mock
    model = LinearModel(mass, nsamp)
    initial = np.zeros(len(sN)) + nsamp.sum()/mass.sum()/len(sN)
    initial = initial * np.random.uniform(1,0.001, size=len(initial))
    hsampler = run_hmc(initial, model, length=100, nsegmax=20,
                       iterations=5000, adapt_iterations=100)

    ptiles = np.percentile(hsampler.chain, [16, 50, 84], axis=0)
    median, minus, plus = ptiles[1,:], ptiles[1,:] - ptiles[0,:], ptiles[2,:] - ptiles[1,:]
    maxapost = hsampler.lnprob.argmax()
    yerr = np.array([minus, plus])

    fig, ax = pl.subplots()
    ax.plot(sN, '-o', label='SPS prediction')
    ax.errorbar(np.arange(len(median)), median, yerr=yerr, color='k', fmt='--o',
                elinewidth=2, capthick=2.0, label='Mock Inferred')
    ax.legend(loc=0)
    ax.set_title(cloud)
    fig.show()
