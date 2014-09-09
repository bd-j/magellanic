import numpy as np
import matplotlib.pyplot as pl
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
from datautils import *
from lfutils import *

imdir = '/Users/bjohnson/Projects/magellanic/images/'
imnamemap = {}
imnamemap['lmc'] = {'IRAC4': imdir+'SAGE_LMC_IRAC8.0_2_mosaic.fits',
                    'IRAC2': imdir+'SAGE_LMC_IRAC4.5_2_mosaic.fits'}
imnamemap['smc'] = {'IRAC4': imdir+'SAGE_SMC_IRAC8.0_2_mosaic.fits',
                    'IRAC2': imdir+'SAGE_SMC_IRAC4.5_2_mosaic.fits'}

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


def brightobjs(cloud, band, mlim=None, nbright=None):
    defstring = cloud_corners(cloud)
    region = ds9reg.Polygon(defstring)
    cat, cols = cloud_cat(cloud)
    subcat = select(cat, cols, region, codes=cols['agb_codes'])
    mag = subcat[cols[band]]
    if mlim is not None:
        bright = mag < mlim
    if nbright is not None:
        oo = np.argsort(mag)
        bright = oo[:nbright]
    return subcat[bright], cols

def get_stamp(imagename, ra, dec, sx=31, sy=31):
    """
    Given lists of RA and Dec (in decimal degrees), and an image name,
    return an array of small images roughly centered on the objects.
    """
    stamps = np.zeros([ len(ra), sx, sy])
    #read image
    im = pyfits.getdata(imagename)
    hdr = pyfits.getheader(imagename)
    wcs = pywcs.WCS(hdr)
    #central values.
    cx, cy = np.round(wcs.wcs_world2pix(ra, dec, 0)).astype('<i8')
    cols = np.arange(sx) - (sx-1)/2
    rows = np.arange(sy) - (sy-1)/2
    #2D array of 1-d indices of a subarray
    patch = (cols[:,None]*im.shape[1] + rows[None,:] )
    #3D array of subarrays for each center
    inds = patch[None,...] + (cy*im.shape[1]+cx)[:,None,None]
    inds = inds.transpose(0,2,1)
    stamps = im.ravel()[inds]
    inds = 0
    return stamps

def magerr_plots(sigma=0.5):
    clfname = 'results_compare/MG08/clf.lmc.irac2.tau10.dat'
    dm = 18.5
    omag, onum = readclf(clfname)
    cmag, cnum = clf_errconv(clfname, sigma=sigma)
    fig, ax = pl.subplots(1, 1)
    ax.plot(omag + dm, onum, linewidth=4,
            label = 'input MG08')
    ax.plot(cmag + dm, cnum, linewidth=4,
            label = 'convolved MG08, $\sigma={0:3.2f}$'.format(sigma))
    ax.set_xlim(14, 4)
    ax.set_yscale('log')
    ax.set_title('LMC @ IRAC2')
    ax.legend(loc=0)
    pl.show()
    fig.savefig('err_convolved_lf_MG08.png')
    pl.close(fig)
    fig, ax = plot_magerrs()
    fig.savefig('obs_mag_errs.png')
    pl.close(fig)
    
if __name__ == '__main__':
    
    cloud, band, nbright = 'smc', 'IRAC4', 16
    imname = imnamemap[cloud.lower()][band]
    bright, cols = brightobjs(cloud, band, nbright=nbright)
    stamps = get_stamp(imname, bright[cols['RA']], bright[cols['DEC']],
                        sx=31, sy=31)
    
    out = open('{0}_{1}_brightest{2}.reg'.format(cloud, band.lower(), nbright),"w")
    nax = int(np.sqrt(nbright))
    fig, axes = pl.subplots(nax, nax, figsize=(8,8))
    for i,b in enumerate(bright):
        ax = axes.flatten()[i]
        out.write('{0:8.4f} {1:8.4f}\n'.format(b[cols['RA']], b[cols['DEC']]))
        ax.imshow(np.log10(stamps[i,:,:]), interpolation='nearest')
        ax.annotate('{0:3.2f}'.format(b[cols[band]]), (23,5), color='red')
        ax.annotate(r'$\alpha,\delta=${0:3.2f},{1:3.2f}'.format(b[cols['RA']], b[cols['DEC']]),
                    (1,30), color='red', size=10)
    [pl.setp(ax.get_xticklabels(), visible = False) for ax in axes.flatten()]
    [pl.setp(ax.get_yticklabels(), visible = False) for ax in axes.flatten()]
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.suptitle('{0} @ {1}'.format(cloud.upper(), band))
    out.close()
    fig.show()
    fig.savefig('{0}_{1}_brightest{2}.png'.format(cloud, band, nbright))
    
