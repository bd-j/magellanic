import numpy as np
import matplotlib.pyplot as pl
from simple_compare import *

def plot_magerrs():
    bands = ['IRAC2', 'IRAC4']
    clouds = ['smc','lmc']
    fig, axes = pl.subplots(2,2)
    # Get the observed CLFs
    for i,cloud in enumerate(clouds):
        defstring = cloud_corners(cloud)
        region = ds9reg.Polygon(defstring)
        cat, cols = cloud_cat(cloud)
        subcat = select(cat, cols, region, codes=cols['agb_codes'])
        for j,band in enumerate(bands):
            ax = axes[j,i]
            ax.set_xlim(4,14)
            ax.set_ylim(0,0.4)
            ax.plot(subcat[cols[band]], subcat[cols[band+'_err']],
                    'o', markersize=5, alpha=0.5, mew=0,
                    label='{0} @ {1}'.format(cloud, band))
            ax.set_xlabel(band)
            ax.set_ylabel(band+'_err')
            #ax.set_title(cloud)
            ax.legend(loc=0)
    return fig, axes

def clf_errconv(clfname, sigma = 0.1):
    mag, num = readclf(clfname)
    nstars = int(num.max())
    sid = np.arange(nstars) + 1
    star_mags = np.interp(sid, num, mag)
    smag_err = star_mags + np.random.normal(0, 1, size = nstars) * sigma
    oo = np.argsort(smag_err)
    return smag_err[oo], sid

def readclf(filename):
    f = open(filename, 'r')
    dat = f.readlines()[2:]
    dat = [d.split() for d in dat]
    data = np.array(dat).astype(float)
    mag, num =  data[:,0], data[:,1]
    good =  np.isfinite(num) & (num > 0)
    mag, num = mag[good], num[good]
    return mag, num

if __name__ == '__main__':
    
    sigma = 0.5
    clfname = 'results_compare/MG08/clf.lmc.irac2.tau10.dat'
    dm = 18.5
    omag, onum = readclf(clfname)
    cmag, cnum = clf_errconv(clfname, sigma=sigma)
    fig, ax = pl.subplots(1, 1)
    ax.plot(omag + dm, onum, label = 'input MG08')
    ax.plot(cmag + dm, cnum, label = 'convolved MG08, $\sigma={0:3.2f}$'.format(sigma))
    ax.set_xlim(14, 4)
    ax.set_yscale('log')
    ax.set_title('LMC @ IRAC2')
    ax.legend(loc=0)
    pl.show()
    fig.savefig('err_convolved_lf_MG08.png')
    fig.close()
    fig, ax = plot_magerrs()
    fig.savefig('obs_mag_errs.png')
    fig.close()
