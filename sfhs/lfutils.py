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
    fstring = '{:.4f}'+ ncol*' {:5.3e}'+'\n'
    out = open(filename,'w')
    out.write('{0}\n mag  {1}\n'.format(lftype, colheads))
    for m,d in zip(mag, dat):
        out.write(fstring.format(m,*d))

    out.close()

def read_villaume_lfs(filename):
    """
    Read a Villaume/FSPS produced cumulative LF file, interpolate LFs
    at each age to a common magnitude grid, and return a dictionary
    containing the interpolated CLFs and ancillary information.

    :param filename:
        The filename (including path) of the Villaume CLF file

    :returns luminosity_func:
        A dictionary with the following key-value pairs:
        ssp_ages: Log of the age for each CLF, ndarray of shape (nage,)
        lf:       The interpolated CLFs, ndarray of shape (nage, nmag)
        bins:     Magnitude grid for the interpolated CLFs, ndarray of
                  shape (nmag,)
        orig:     2-element list consisting of the original magnitude grids
                  and CLFs, each also as lists.
        
    """
    age, bins, lfs = [], [], []
    f = open(filename, "r")
    for i,line in enumerate(f):
        dat = [ float(d) for d in line.split() ]
        if (i % 3) == 0:
            age += [ dat[0]]
        elif (i % 3) == 1:
            bins += [dat]
        elif (i % 3) == 2:
            lfs += [dat]
    f.close()
    
    age = np.array(age)
    minage, maxage = np.min(age)-0.05, np.max(age)+0.10
    minl, maxl = np.min(bins)[0], np.max(bins)[0]+0.01
    allages = np.arange(minage, maxage, 0.05)
    mags = np.arange(minl, maxl, 0.01)
    print(minl, maxl)
    
    lf = np.zeros([ len(allages), len(mags)])
    for i, t in enumerate(allages):
        inds = np.isclose(t,age)
        if inds.sum() == 0:
            continue
        ind = np.where(inds)[0][0]
        x = np.array(bins[ind] + [np.max(mags)])
        y = np.log10(lfs[ind] +[np.max(lfs[ind])])        
        lf[i, :] = 10**interp1d(np.sort(x), np.sort(y), fill_value = -np.inf, bounds_error = False)(mags)

    luminosity_func ={}
    luminosity_func['ssp_ages'] = allages
    luminosity_func['lf'] = lf
    luminosity_func['bins'] = mags
    luminosity_func['orig'] = [bins, lfs]

    return luminosity_func

    
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
