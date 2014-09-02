import sys, pickle, copy
import numpy as np
import matplotlib.pyplot as pl

import astropy.io.fits as pyfits
try: 
    import fsps
except ImportError:
    #you wont be able to predict the integrated spectrum
    # filterlist must be set to None in calls to total_cloud_data
    sps = None
try:
    from sedpy import observate
except ImportError:
    #you won't be able to predict integrated magnitudes
    pass
from sputils import read_lfs
import regionsed as rsed
import mcutils as utils

wlengths = {'2': '{4.5\mu m}',
            '4': '{8\mu m}'}

def total_cloud_data(cloud, basti=False,
                     lfstrings=['z{0:02.0f}_tau10_vega_irac4_lf.txt'],
                     filterlist = None):
    
    #########
    # Initialize the ingredients (SPS, SFHs, LFs)
    #########
    # SPS
    if filterlist is not None:
        sps = fsps.StellarPopulation(add_agb_dust_model = True)
        sps.params['sfh'] = 0
        sps.params['agb_dust'] = agb_dust
        dust = ['nodust', 'agbdust']
        sps.params['imf_type'] = 0.0 #salpeter
        filterlist = observate.load_filters(filterlist)
    
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
    # these are sored as a list of lists.  The outer list is differnt
    # isochrones, wavelengths, agb_dusts, the inner list if different
    # metallicities
    lf_bases = []
    for lfstring in lfstrings:
        print(lfstring)
        lffiles = [lfstring.format(z) for z in zlist]
        lf_bases += [[read_lfs(f) for f in lffiles]]
    
    #############
    # Loop over each region, do SFH integrations, filter convolutions
    # and populate output images and LF cubes
    ############
    total_sfhs = None
    bins = rsed.lfbins
    for n, dat in regions.iteritems():
        total_sfhs = sum_sfhs(total_sfhs, dat['sfhs'])
        total_zmet = dat['zmet']
    
    #loop over the different bands (and whatever else) for the LFs
    lfs, maggies = [], None
    for i,lf_base in enumerate(lf_bases):
        lf = rsed.one_region_lf(copy.deepcopy(total_sfhs),
                                total_zmet, lf_base)
        lfs += [lf]
    if filterlist is not None:
        spec, wave = rsed.one_region_sed(copy.deepcopy(total_sfhs),
                                         total_zmet, sps )
        mags = observate.getSED(wave, spec * rsed.to_cgs,
                                filterlist = filterlist)
        maggies = 10**(-0.4 * np.atleast_1d(mags))
    
    #############
    # Write output
    ############
    total_values = {}
    total_values['agb_clfs'] = lfs
    total_values['clf_mags'] = bins
    total_values['sed_ab_maggies'] = maggies
    total_values['sed_filters'] = filters
    total_values['lffiles'] = lfstrings
    
    return total_values, total_sfhs

def sum_sfhs(sfhs1, sfhs2):
    """
    Accumulate individual sets of SFHs into a total set of SFHs.  This
    assumes that the individual SFH sets all have the same number and
    order of metallicities, and the same time binning.
    """
    if sfhs1 is None:
        return copy.deepcopy(sfhs2)
    elif sfhs2 is None:
        return copy.deepcopy(sfhs1)
    else:
        out = copy.deepcopy(sfhs1)
        for s1, s2 in zip(out, sfhs2):
            s1['sfr'] += s2['sfr']
        return out
    
def plot_lf(base, wave, lffile):
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


def write_clf(wclf, filename, lftype):
    """
    Given a 2 element list decribing the CLF, write it to `filename'.
    """
    out = open(filename,'w')
    out.write('{0}\n mag  N<m\n'.format(lftype))
    for m,n in zip(wclf[0], wclf[1]):
        out.write('{0:.4f}   {1}\n'.format(m,n))
    out.close()

def write_composite_clfs(total_values, ldir, rdir):
    """
    Take the 0th element output of total_cloud_data and write the
    composite CLFs it contains to .dat files
    """
    dat = total_values
    for i, lfstring in enumerate(dat['lffiles']):
        outfile = lfstring.replace(ldir, rdir).replace('z{0:02.0f}_','').replace('.txt','.dat')
        write_clf([dat['clf_mags'], dat['agb_clfs'][i]], outfile, lfstring)

        
if __name__ == '__main__':
    
    #filters = ['galex_NUV', 'spitzer_irac_ch1',
    #           'spitzer_irac_ch4', 'spitzer_mips_24']
    filters = None
    
    ldir, cdir = 'lf_data/', 'composite_lfs/'
    outst = '{0}_cg10.p'
    # total_cloud_data will loop over the appropriate (for the
    # isochrone) metallicities for a given lfst filename template
    lfst = '{0}z{{0:02.0f}}_tau{1:2.0f}_vega_irac{2}_lf.txt'
    basti = False
    
    
    for cloud in ['smc', 'lmc']:
        lfstrings = []
        for agb_dust in [1.0]:
            for band in ['2','4']:
                lfstrings += [lfst.format(ldir, agb_dust*10.0, band)]
        print(cloud)
        dat = total_cloud_data(cloud, lfstrings=lfstrings, basti=basti,
                               filterlist=filters)
        out = open(outst.format(cloud), "wb")
        pickle.dump(dat[0], out)
        out.close()

        write_composite_clfs(dat[0], ldir, '{0}cclf_{1}_'.format(cdir, cloud))

        
