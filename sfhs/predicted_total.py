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
from lfutils import *

wlengths = {'2': '{4.5\mu m}',
            '4': '{8\mu m}'}

dmod = {'smc':18.9,
        'lmc':18.5}
    
def total_cloud_data(cloud, filternames = None, basti=False,
                     lfstring='z{0:02.0f}_tau10_vega_irac4_lf.txt',
                     agb_dust=1.0,
                     one_metal=None):
    
    #########
    # Initialize the ingredients (SPS, SFHs, LFs)
    #########
    # SPS
    if filternames is not None:
        sps = fsps.StellarPopulation(add_agb_dust_model = True)
        sps.params['sfh'] = 0
        sps.params['agb_dust'] = agb_dust
        dust = ['nodust', 'agbdust']
        sps.params['imf_type'] = 0.0 #salpeter
        filterlist = observate.load_filters(filternames)
    else:
        filterlist = None
    # SFHs
    if cloud.lower() == 'lmc':
        regions = utils.lmc_regions()
        nx, ny = 48, 38
        zlist = [7, 11, 13, 16]
        if basti:
            zlist = [3,4,5,6]
    elif cloud.lower() == 'smc':
        regions = utils.smc_regions()
        nx, ny = 20, 23
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
    
    lffiles = [lfstring.format(z) for z in zlist]
    lf_base = [read_lfs(f) for f in lffiles]
    
    #############
    # Loop over each region, do SFH integrations, filter convolutions
    # and populate output images and LF cubes
    ############
    total_sfhs = None
    bins = rsed.lfbins
    for n, dat in regions.iteritems():
        total_sfhs = sum_sfhs(total_sfhs, dat['sfhs'])
        total_zmet = dat['zmet']

    #collapse SFHs and LFs base to one metallicity
    if one_metal is not None:
        ts = None
        for sfh in total_sfhs:
            ts = sum_sfhs(ts, sfh)
        lf_base = [lfbase[one_metal]] 
        total_zmet = [total_zmet[one_metal]]
        
    #get LFs broken out by age and metallicity as well as the total
    lfs_zt, lf, logages = rsed.one_region_lfs(copy.deepcopy(total_sfhs),
                                              total_zmet, lf_base)
    maggies, mass = None, None
    if filterlist is not None:
        spec, wave, mass = rsed.one_region_sed(copy.deepcopy(total_sfhs),
                                         total_zmet, sps )
        mags = observate.getSED(wave, spec * rsed.to_cgs,
                                filterlist = filterlist)
        maggies = 10**(-0.4 * np.atleast_1d(mags))
    
    #############
    # Write output
    ############
    total_values = {}
    total_values['agb_clf'] = lf
    total_values['agb_clfs_zt'] = lfs_zt
    total_values['clf_mags'] = bins
    total_values['logages'] = logages
    total_values['sed_ab_maggies'] = maggies
    total_values['sed_filters'] = filternames
    total_values['lffile'] = lfstring
    total_values['mstar'] = mass
    total_values['zlist'] = zlist
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
    
if __name__ == '__main__':
    
    #filters = ['galex_NUV', 'spitzer_irac_ch1',
    #           'spitzer_irac_ch4', 'spitzer_mips_24']
    filters = None
    
    ldir, cdir = 'lf_data/', 'composite_lfs/'
    outst = '{0}_n2teffcut.p'
    # total_cloud_data will loop over the appropriate (for the
    # isochrone) metallicities for a given lfst filename template
    lfst = '{0}z{{0:02.0f}}_tau{1:2.1f}_vega_irac{2}_n2_teffcut_lf.txt'
    basti = False
    agb_dust=1.0
    
    #loop over clouds (and bands and agb_dust) to produce clfs
    for cloud in ['smc', 'lmc']:
        rdir = '{0}cclf_{1}_'.format(cdir, cloud)
        for band in ['2','4']:
            lfstring = lfst.format(ldir, agb_dust, band)
            dat, sfhs = total_cloud_data(cloud, filternames=filters, agb_dust=agb_dust,
                                         lfstring=lfstring, basti=basti)
            outfile = lfstring.replace(ldir, rdir).replace('z{0:02.0f}_','').replace('.txt','.dat')
            write_clf_many([dat['clf_mags'], dat['agb_clf']], outfile, lfstring)
            
            #fig, ax = plot_weighted_lfs(dat, agebins = np.arange(9)*0.25 + 7.75, dm=dmod[cloud])
            #fig.suptitle('{0} @ IRAC{1}'.format(cloud.upper(), band))
            #fig.savefig('{0}_clfs_by_age_and_Z_irac{1}'.format(cloud, band))
            #pl.close(fig)

            agebins = np.arange(9)*0.25 + 7.75
            tbinned_lfs = [rebin_lfs(lf, ages, agebins) for
                           lf, ages in zip(dat['agb_clfs_zt'], dat['logages'])]
            
            print(cloud, dat['mstar'])
        
