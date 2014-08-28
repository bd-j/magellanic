import sys, pickle
import numpy as np
import matplotlib.pyplot as pl

import astropy.io.fits as pyfits
import fsps
from sedpy import observate

from sputils import read_lfs
import regionsed as rsed
import mcutils as utils

wlengths = {'2': '{4.5\mu m}',
            '4': '{8\mu m}'}

def total_cloud_data(cloud, agb_dust, lf_bands, out='total_data.p',
                     lfstrings=['z{0:02.0f}_tau10_vega_irac4_lf.txt']):
    
    #Run parameters
    filters = ['galex_NUV', 'spitzer_irac_ch1',
               'spitzer_irac_ch4', 'spitzer_mips_24']
    
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
        total_sfhs = add_sfhs(total_sfhs, dat['sfhs'])
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
    total_values['lffiles'] = lfstrings
    out = open(out, "wb")
    pickle.dump(total_values, out)
    out.close()
    return total_values

def add_sfhs(sfhs1, sfhs2):
    """
    Accumulate individual sets of SFHs into a total set of SFHs.  This
    assumes that the individual SFH sets all have the same number and
    order of metallicities, and the same time binning.
    """
    if sfhs1 is None:
        return sfhs2.copy()
    elif sfhs2 is None:
        return sfhs1.copy()
    else:
        out = sfhs1.copy()
        for s1, s2 in zip(out, sfhs2):
            s1['sfr'] += s2['sfr']
        return out
    
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

    ldir, outdir = 'lf_data/'
    out = 'results_predicted/padova.p'
    st = '{0}z{{0:02.0f}}_tau{1}_vega_irac{2}_lf.txt'
    for cloud in ['smc', 'lmc']:
        for agb_dust in [1.0]:
            for band in ['2','4']:
                lfstrings += [st.format(ldir, agb_dust*10.0, band)]
        dat = total_cloud_data(cloud, lfstrings, out)
