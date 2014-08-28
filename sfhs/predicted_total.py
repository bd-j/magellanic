import sys, pickle
import numpy as np
import matplotlib.pyplot as pl

import astropy.io.fits as pyfits
import fsps
from sedpy import observate

from sfhutils import read_lfs
import regionsed as rsed
import mcutils as utils


wlengths = {'2': '{4.5\mu m}',
            '4': '{8\mu m}'}

def total_lf(cloud, agb_dust, lf_bands,
             lfstrings=['z{0:02.0f}_tau10_vega_irac4_lf.txt']):
    
    #Run parameters
    filters = ['galex_NUV', 'spitzer_irac_ch1', 'spitzer_irac_ch4', 'spitzer_mips_24']
    min_tpagb_age = 0.0
    ldir, outdir = 'lf_data/', 'results_predicted/'
    
    #########
    # Initialize the ingredients (SPS, SFHs, LFs)
    #########
    # SPS
    sps = fsps.StellarPopulation(add_agb_dust_model = True)
    sps.params['sfh'] = 0
    sps.params['agb_dust'] = agb_dust
    dust = ['nodust', 'agbdust']
    sps.params['imf_type'] = 0.0 #salpeter

    filterlist = observate.load_filters(filters)
    
    # SFHs
    if cloud.lower() == 'lmc':
        regions = utils.lmc_regions()
        nx, ny, dm = 48, 38, 18.5
        zlist = [7, 11, 13, 16]
        if basti:
            zlist = [3,4,5,6]

    elif cloud.lower() == 'smc':
        regions = utils.smc_regions()
        nx, ny, dm = 20, 23, 18.9
        zlist = [7, 13, 16]
        if basti:
            zlist = [3,5,6]

    else:
        print('do not understand your MC designation')
        
    rheader = regions.pop('header') #dump the header info from the reg. dict        
    
    # LFs
    lf_bases = []
    for lfstring in lfstrings:
        lffiles = [lfstring.format(z) for z in zlist]
        lf_bases += [[read_lfs(f) for f in lffiles]]
    
    #############
    # Loop over each region, do SFH integrations, filter convolutions
    # and populate output images and LF cubes
    ############
    total_sfhs = None
    bins = rsed.lbins
    for n, dat in regions.iteritems():
        total_sfhs = accumulate_sfhs(total_sfhs, dat['sfhs'])
        total_zmet = dat['zmet']

    lfs = []
    #loop over the differnt bands (and whatever else) for the LFs
    for lf_base in lf_bases:
        spec, lf, wave = rsed.one_region_sed(total_sfhs, total_zmet,
                                             sps, lf_bases=lf_base)
        lfs += [lf]
        
    mags = observate.getSED(wave, spec * rsed.to_cgs,
                            filterlist = filterlist)
    maggies = 10**(-0.4 * np.atleast_1d(mags))
    
    #############
    # Write output
    ############
    
    total_values = {}
    total_values['agb_clfs'] = lfs
    total_values['clf_mags'] = bins
    total_values['clf_bands'] = lf_bands
    total_values['sed_ab_maggies'] = maggies
    total_values['sed_filters'] = filters
    out = open("{0}total_data.{1}.tau{2:02.0f}.p".format(outdir, cloud.lower(), agb_dust*10), "wb")
    pickle.dump(total_values, out)
    out.close()
    return total_values

def accumulate_sfhs(total, one_set):
    """
    Accumulate individual sets of SFHs into a total set of SFHs.  This
    assumes that the individual SFH sets all have the same number and
    order of metallicities, and the same time binning.
    """
    if total is None:
        return one_set
    else:
        for t, o in zip(total, one_set):
            t['sfr'] += o['sfr']
        return total
    
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
    for cloud in ['smc', 'lmc']:
        for agb_dust in [0.5, 1.0]:
                total_lf(cloud, agb_dust, ['2', '4'])
        #main(cloud, 1.0, '4')
        #main(cloud, 0.5, '2')
