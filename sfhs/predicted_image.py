import sys, pickle
import numpy as np
import matplotlib.pyplot as pl

import astropy.io.fits as pyfits
import fsps
from sedpy import observate

from sfhutils import read_lfs
import regionsed as rsed
import mcutils as utils


def main():
    #Run parameters
    cloud, filters = 'smc', ['galex_NUV', 'spitzer_irac_ch1', 'spitzer_irac_ch4', 'spitzer_mips_24']
    min_tpagb_age, lf_band, wave, agb_dust = 0.0, '4', '{8\mu m}', 1.0
    ldir, outdir = 'lf_data/', 'test/'
    
    #########
    # Initialize the ingredients (SPS, SFHs, LFs)
    #########
    # SPS
    sps = fsps.StellarPopulation(add_agb_dust_model = True)
    sps.params['agb_dust'] = agb_dust
    dust = ['nodust', 'agbdust']
    sps.params['imf_type'] = 0.0

    filterlist = observate.load_filters(filters)
    
    # SFHs
    if cloud.lower() == 'lmc':
        regions = utils.lmc_regions()
        nx, ny, dm = 48, 38, 18.5
        zlist = [7, 11, 13, 16]
    elif cloud.lower() == 'smc':
        regions = utils.smc_regions()
        nx, ny, dm = 20, 23, 18.9
        zlist = [7, 13, 16]
    else:
        print('do not understand your MC designation')
        
    fstring = '{0}z{1:02.0f}_tau{2:02.0f}_vega_irac{3}_lf.txt'
    lffiles = [fstring.format(ldir, z, agb_dust*10, lf_band) for z in zlist]
    rheader = regions.pop('header') #dump the header info from the reg. dict
    
    # LFs
    try:
        lf_bases = [read_lfs(f) for f in lffiles]
        #zero out select ages
        for j, base in enumerate(lf_bases):
            blank = base['ssp_ages'] <= min_tpagb_age
            base['lf'][blank,:] = 0
            plot_lf(base, wave, lffiles[j])

    except(NameError):
        lf_bases = None
    
    #############
    # Loop over each region, do SFH integrations, filter convolutions
    # and populate output images and LF cubes
    ############
    
    im = np.zeros([ len(filters), nx, ny]) #flux in each band in each region
    bins = rsed.lbins
    agb = np.zeros([ len(bins), nx, ny]) #cumulative LF in each region
    for n, dat in regions.iteritems():
        spec, lf, wave = rsed.one_region_sed(dat['sfhs'], dat['zmet'],
                                             sps, lf_bases=lf_bases)
        mags = observate.getSED(wave, spec * rsed.to_cgs,
                                filterlist = filterlist)
        maggies = 10**(-0.4 * np.atleast_1d(mags))
        x, y = utils.regname_to_xy(n, cloud=cloud)
        agb[:, x, y] = lf[:, None] / np.size(x)
        im[:, x, y] = maggies[:, None]/ np.size(x)

    #############
    # Write output
    ############
    for i, f in enumerate(filters):
        write_image(im[i,:,:].T, cloud, f, outdir=outdir, agb_dust=agb_dust)
    
    #write out AGB N(>M) images as fits
    #for lim in np.arange(-6.7, -9.5, -0.5):
    #    ind = np.searchsorted(bins, lim) -1
    #    pyfits.writeto('test.fits', agb[ind,:,:].T, clobber = True)

    #write out AGB N(>M) images as a pickle file
    agb_cube = {}
    agb_cube['agb_clf_cube'] = agb
    agb_cube['mag_bins'] = bins
    out = open("{0}clf.{1}.tau{2:02.0f}.irac{3}.p".format(outdir, cloud.lower(), agb_dust*10, lf_band), "wb")
    pickle.dump(agb_cube, out)
    out.close()

    # Plot the total LF
    fig, ax = pl.subplots(1,1)
    lf_tot = agb.sum(-1).sum(-1)
    ax.plot(bins + dm, lf_tot)
    ax.set_ylabel(r'$N(<M)$ (total for cloud)')
    ax.set_xlabel(r'$m_{}$ (Vega apparent)'.format(wave))
    ax.set_title(cloud.upper())
    ax.set_yscale('log')
    fig.savefig('{0}total_agb_clf.{1}.tau{2:02.0f}.irac{3}.png'.format(outdir, cloud.lower(), agb_dust*10, lf_band))

def write_image(im, cloud, filtname, outdir='./', agb_dust=1.0):
    """
    Write out images as fits and jpg
    """
    fig, ax = pl.subplots(1,1)
    image = ax.imshow(np.log10(im), interpolation = 'nearest', origin = 'lower')
    ax.set_title('{0}  @ {1}'.format(cloud.upper(), filtname))
    ax.set_xlabel('RA (pixels)')
    ax.set_ylabel('Dec (pixels)')
    cbar = pl.colorbar(image, orientation = 'horizontal', shrink = 0.7, pad = 0.12)
    cbar.set_label(r'log F({0}) (AB maggies)'.format(filtname))
    fstring = '{0}hzpred.{1}.tau{2:02.0f}.{3}'.format( outdir, cloud, agb_dust*10, filtname)
    pl.savefig(fstring + '.png')
    pl.close(fig)
    pyfits.writeto(fstring + '.fits', im, clobber = True)

def plot_lf(base, wave, lffile):
    """
    Plot the interpolated input lfs to make sure they are ok
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


if __name__ == '__main__':
    main()
