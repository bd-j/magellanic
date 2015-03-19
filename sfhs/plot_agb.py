import numpy as np
import pickle, acor, triangle, sys
import matplotlib.pyplot as pl

badenes_file = 'tex/badenes_results/LMC_MCMC_DTD_AGB_Unbinned_Iter000.dat'
filename = 'lmc_All_wbias_chain.p'
with open(filename) as f:
    result = pickle.load(f)
    
bdat = np.loadtxt(badenes_file, skiprows=1)


fig, axes = pl.subplots(4,4, sharex=True, figsize=(12, 8))
for i in range(len(bdat)):
    ax = axes.flatten()[i]
    ax.hist(result['chain'][:,i], bins=50, label='BDJ',
            histtype='stepfilled', alpha=0.5)
    ax.axvline(bdat[i, 3], label='CB', color='red')
    ax.axvline(bdat[i, 4], linestyle='--', color='red')
    ax.axvline(bdat[i, 5], linestyle='--', color='red')
    ax.axvline(bdat[i, 6], linestyle=':', color='red')
    ax.axvline(bdat[i, 7], linestyle=':', color='red')
    ax.set_xlabel(r'$\theta$({0}-{1})'.format(bdat[i,1], bdat[i,2]))
    ax.set_xlim(0, 2e-4)
#
fig.show()
pl.gcf().set_tight_layout(True)

efig, eaxes = pl.subplots(4,4, sharex=True, figsize=(12, 8))
for i in range(len(bdat)):
    ax = eaxes.flatten()[i]
    tau = acor.acor(result['chain'][:,i])[0]
    ax.plot(result['chain'][:,i])
    ax.text(0.1, 0.85, r'$\theta_{{{0}}}, \tau={1:4.1f}$'.format(i, tau),
            bbox=dict(facecolor='white', edgecolor='black', pad=10.0),
            transform=ax.transAxes, fontsize=6)


efig.show()
pl.gcf().set_tight_layout(True)

sys.exit()

ffig, faxes = pl.subplots(1,2, figsize=(10,4))
theta = hsampler.chain[maxapost,:]
dt = 10**(esfh['t2']) - 10**(esfh['t1'])
faxes[0].plot(time[:-1], theta[:-1]*dt, '-o')
faxes[0].set_ylabel(r'$\#/(M_\odot/yr)_j$')
faxes[1].plot(time[:-1], theta[:-1]*mass.sum(axis=1)[:-1]/N.sum(), '-o')
faxes[1].set_ylabel(r'fraction of catalog due to $t_j$')
[ax.set_xlabel(r'$t_j$') for ax in faxes]
ffig.show()

clr = 'darkcyan'
bfig, baxes = pl.subplots(figsize=(12,8))
bp = baxes.boxplot(hsampler.chain,  labels = [str(t) for t in time],
                   whis=[16, 84], widths=0.9,
                   boxprops = {'alpha': 0.3, 'color':clr},
                   whiskerprops = {'linestyle':'-', 'linewidth':2, 'color':'black'},
                   showcaps=False, showfliers=False, patch_artist=True)
baxes.set_xlabel(r'$\log \, t_j ($yrs$)$', labelpad=15)
baxes.set_ylabel(r'$\theta_j \, (AGB \#/M_\odot) \, ($specific frequency$)$')
baxes.set_title(cloud.upper()+extralabel)
bfig.show()
bfig.savefig('{0}{1}_theta.pdf'.format(cloud.lower(), extralabel) )


try:
    import pandas as pd
    import seaborn as sns
    chain = result['chain']
    chain = pd.DataFrame(chain,
                         columns=[str(i) for i in range(chain.shape[1])])
    g = sns.PairGrid(chain)
    g.map_lower(sns.kdeplot, cmap="Blues_d", bw='silverman',legend=False)
    g.map_diag(pl.hist, lw=3, legend=False)
    g.fig.set_figwidth(30)
    g.fig.set_figheight(30)
    g.savefig('{0}{1}_corner.pdf'.format(cloud.lower(), extralabel) )
except(ImportError):
    pass
