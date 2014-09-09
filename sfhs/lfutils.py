import numpy as np
import matplotlib.pyplot as pl

def clf_to_lf(clfname, bins=None):
    mag, num = readclf(clfname)
    dn = -np.diff(num)
    if bins is not None:
        num = np.interp(bins, num, mag)
        dn = -np.diff(num)
        mag = bins
        
    return dn, mag

def rebin_lfs(lf, logages, agebins):
    lf_rebinned = np.zeros([ len(agebins), lf.shape[1]])
    for i, (start, stop) in enumerate(zip(agebins[0:-1], agebins[1:])):
        this = (logages <= stop) & (logages > start)
        if this.sum() == 0:
            continue
        lf_rebinned[i,:] = lf[this,:].sum(axis=0)
    return lf_rebinned

def readclf(filename):
    f = open(filename, 'r')
    dat = f.readlines()[2:]
    dat = [d.split() for d in dat]
    data = np.array(dat).astype(float)
    mag, num =  data[:,0], data[:,1]
    good =  np.isfinite(num) & (num > 0)
    mag, num = mag[good], num[good]
    return mag, num
 
def write_clf(wclf, filename, lftype, colheads='N<m'):
    """
    Given a 2 element list decribing the CLF, write it to `filename'.
    """
    out = open(filename,'w')
    out.write('{0}\n mag  {1}\n'.format(lftype, colheads))
    for m,n in zip(wclf[0], wclf[1]):
        out.write('{0:.4f}   {1}\n'.format(m,n))
    out.close()

def write_clf_many(clf, filename, lftype, colheads='N<m'):
    mag, dat = clf
    dat = np.atleast_2d(dat).T
    nrow = len(mag)
    assert nrow == dat.shape[0]
    ncol = dat.shape[1]
    fstring = '{0:.4f}'+ ncol*'{}'+'\n'
    out = open(filename,'w')
    out.write('{0}\n mag  {1}\n'.format(lftype, colheads))
    for m,d in zip(mag, dat):
        out.write(fstring.format(m,*d))

    out.close()

def plot_ssp_lf(base, wave, lffile):
    """
    Plot the interpolated input lfs to make sure they are ok.
    """
    ncolors = base['lf'].shape[0]
    cm = pl.get_cmap('gist_rainbow')
    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1.*i/ncolors) for i in range(ncolors)])
    for i,t in enumerate(base['ssp_ages']):
        ax.plot(base['bins'], base['lf'][i,:], linewidth = 3,
                label = '{:4.2f}'.format(t), color = cm(1.*i/ncolors))
    ax.legend(loc =0, prop = {'size':6})
    ax.set_ylim(1e-6,3e-4)
    ax.set_yscale('log')
    ax.set_xlabel(r'$M_{}$'.format(wave))
    ax.set_ylabel(r'$n(<M, t)$')
    fig.savefig('{}.png'.format(lffile.replace('.txt','')))
    pl.close(fig)

def plot_weighted_lfs(total_values, agebins=None, dm=0.0):
    cm = pl.get_cmap('jet')

    lfzt = total_values['agb_clfs_zt']
    mags = total_values['clf_mags']
    total_lf = total_values['agb_clf']
    logages = total_values['logages']
    zlist = total_values['zlist']
    nz = len(zlist)
    ncols = nrows = np.ceil(np.sqrt(nz*1.0)).astype('<i8')
    fig, axes = pl.subplots(nrows, ncols, figsize=(10,8)) 
    for (z, lf, ages, ax) in zip(zlist, lfzt, logages, axes.flatten()):
        ax.plot(mags+dm, total_lf, color='k', label='total CLF',
                linewidth=5)
        if agebins is None:
            agebins = ages
        ncolors = len(agebins)-1
        ax.set_color_cycle([cm(1.*i/ncolors) for i in range(ncolors)])
        wlf = rebin_lfs(lf, ages, agebins)
        for i, (start, stop) in enumerate(zip(agebins[0:-1], agebins[1:])):
            ax.plot(mags+dm, wlf[i,:], label = '{0}<logt<{1}'.format(start, stop),
                    linewidth=3, color = cm(1.*i/ncolors))
        ax.legend(loc=0, prop = {'size':6})
        ax.set_yscale('log')
        ax.set_xlim(12,4)
        ax.set_title('Z={0}'.format(z))
        ax.set_ylim(1,1e6)
    return fig, axes
