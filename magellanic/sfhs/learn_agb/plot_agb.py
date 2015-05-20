import numpy as np
import pickle, acor, triangle, sys
import matplotlib.pyplot as pl
from magellanic import sps_freq as sfreq
from scipy.special import erf

def plot_pdfs(result, bdat=None):
    """ PDFs for theta
    """
    fig, axes = pl.subplots(4,4, sharex=True, figsize=(12, 8))
    for i in range(len(bdat)):
        ax = axes.flatten()[i]
    #    ax.hist(result2['chain'][2500:,i], bins=50, label='BDJ',
    #            histtype='stepfilled', alpha=0.5)
        ax.hist(result['chain'][2500:,i], bins=50, label='BDJ w/CB coo',
                histtype='stepfilled', alpha=0.5)
        if bdat is not None:
            ax.axvline(bdat[i, 3], label='CB', color='red')
            ax.axvline(bdat[i, 4], linestyle='--', color='red')
            ax.axvline(bdat[i, 5], linestyle='--', color='red')
            ax.axvline(bdat[i, 6], linestyle=':', color='red')
            ax.axvline(bdat[i, 7], linestyle=':', color='red')
            ax.set_xlabel(r'$\theta$({0}-{1})'.format(bdat[i,1], bdat[i,2]))
        ax.set_xlim(0, 2e-4)
    fig.axes[0].legend(loc=0, prop={'size':8})
    fig.show()
    pl.gcf().set_tight_layout(True)
    return fig, axes

def plot_chain(result, **kwargs):
    """ Chain evolution
    """
    efig, eaxes = pl.subplots(4,4, sharex=True, figsize=(12, 8))
    for i in range(result['chain'].shape[-1]):
        ax = eaxes.flatten()[i]
        tau = acor.acor(result['chain'][:,i])[0]
        ax.plot(result['chain'][:,i])
        ax.text(0.1, 0.85, r'$\theta_{{{0}}}, \tau={1:4.1f}$'.format(i, tau),
                bbox=dict(facecolor='white', edgecolor='black', pad=10.0),
                transform=ax.transAxes, fontsize=6)


    efig.show()
    pl.gcf().set_tight_layout(True)
    return efig, eaxes

def plot_theta_time(result, boxplot=False, bdat=None,
                    clr = 'darkcyan', **kwargs):
    """ Theta vs time with uncertainties
    """
    
    bfig, baxes = pl.subplots(figsize=(12,8))
    x = np.arange(result['chain'].shape[-1])
    if boxplot:
        bp = baxes.boxplot(result['chain'],  labels = [str(t) for t in result['time']],
                           whis=[16, 84], widths=0.9,
                           boxprops = {'alpha': 0.3, 'color':clr},
                           whiskerprops = {'linestyle':'-', 'linewidth':2, 'color':'black'},
                           showcaps=False, showfliers=False, patch_artist=True)
    else:
        p = np.percentile(result['chain'],[16,50,84], axis=0)
        
        yerr = np.array([p[1]-p[0], p[2]-p[1]])
        baxes.errorbar(x, p[1], yerr=yerr, color=clr, fmt='--o',
                       elinewidth=2, capthick=2.0, label='BDJ')
        baxes.set_xticks(x)
        baxes.set_xticklabels([str(t) for t in result['time']])

    print(bdat is None)
    if bdat is not None:
        yerr = np.array([bdat[:,3]-bdat[:,4], bdat[:,5]-bdat[:,3]])
        baxes.errorbar(x, bdat[:,3], yerr=yerr, color='red', fmt='--o',
                       elinewidth=2, capthick=2.0, label='CB')
        
    baxes.set_xlabel(r'$\log \, t_j ($yrs$)$', labelpad=15)
    baxes.set_ylabel(r'$\theta_j \, (AGB \#/M_\odot) \, ($specific frequency$)$')
    #baxes.set_title(result['cloud'].upper()+extralabel)
    #bfig.show()
    return bfig, baxes

def overplot_sps(result, fax, stype='cmd', **kwargs):
    """Overplot sps predictions
    """
    bfig, baxes = fax
    if stype == 'cmd':
        sfunc = sfreq.agb_select_function_cmd
    elif stype == 'phase':
        sfunc = sfreq.agb_select_function
    else:
        sfunc = stype
        
    agbtype = {'MG08':0,'CG10':1,'VCJ':2}
    for aname, atype in agbtype.iteritems():
        nex, zact = sfreq.make_freq_prediction(result['cloud'], result['esfh'],
                                            select_function=sfunc,
                                            tpagb_norm_type=atype, **kwargs)
        baxes.plot(nex, '-o', label=aname)
    return bfig, baxes


def plot_catfrac(result):
    ffig, faxes = pl.subplots(1,2, figsize=(10,4))
    maxapost = result['lnprob'].argmax()
    theta = result['chain'][maxapost,:]
    esfh, time, mass = result['esfh'], result['time'], result['mass']
    dt = 10**(esfh['t2']) - 10**(esfh['t1'])
    faxes[0].plot(time[:-1], theta[:-1]*dt, '-o')
    faxes[0].set_ylabel(r'$\#/(M_\odot/yr)_j$')
    faxes[1].plot(time[:-1], theta[:-1]*mass.sum(axis=1)[:-1]/result['N'].sum(), '-o')
    faxes[1].set_ylabel(r'fraction of catalog due to $t_j$')
    [ax.set_xlabel(r'$t_j$') for ax in faxes]
    #ffig.show()
    return ffig, faxes

def gaussfit_lnprob(theta, samples=0):
    """Not working
    """
    mu, sigma = theta.tolist()
    if (mu < 0) or (sigma < 0):
        return -np.infty
    #normalizing constant for the constraint that x be greater than 0.
    Ainv = sigma * (1 + erf(mu/(sigma*np.sqrt(2)))) * np.sqrt(np.pi/2.0)
    lnp = -0.5 * ((samples-mu)**2/(sigma**2) + 2.0 * np.log(Ainv))
    return lnp


if __name__ == "__main__":
    
    badenes_file = 'tex/badenes_results/LMC_MCMC_DTD_AGB_Unbinned_Iter000.dat'
    bdat = np.loadtxt(badenes_file, skiprows=1)

    import fsps
    sps = fsps.StellarPopulation(compute_vega_mags=True)
    sps.params['sfh'] = 0
    sps.params['imf_type'] = 0
    sps.params['tpagb_norm_type'] = 2 #VCJ
    sps.params['add_agb_dust_model'] = True
    sps.params['agb_dust'] = 1.0

    #filename = 'chains/lmc_All_cb_noRS_chain.p'
    #filename = 'chains/smc_All_chain.p'
    filename = 'chains/lmc_CX_chain.p'
    with open(filename) as f:
        result = pickle.load(f)
    
    def select(isoc_dat, **kwargs):
        c, o, boyer, xagb = sfreq.boyer_cmd_classes(isoc_dat, **kwargs)    
        return isoc_dat[c | xagb]
        
    #fig, fax = plot_pdfs(result, bdat=bdat)
    #efig, eax = plot_chain(result)
    bfig, bax = plot_theta_time(result, #bdat=bdat,
                                clr='black', sps=sps)
    bfig, bax = overplot_sps(result, (bfig, bax), stype=select)
    bax.legend(loc=0)
    bfig.show()
