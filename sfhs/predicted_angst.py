import sys, pickle, copy
import numpy as np
import matplotlib.pyplot as pl

import astropy.io.fits as pyfits
from sputils import read_lfs
import regionsed as rsed
from lfutils import *
import fsps
from sedpy import observate
from sfhutils import load_angst_sfh

wlengths = {'2': '{4.5\mu m}',
            '4': '{8\mu m}'}


def total_galaxy_data(sfhfilename, zindex, filternames = None, basti=False,
                     lfstring=None, agb_dust=1.0):

    total_sfhs = load_angst_sfh(sfhfilename)
    zlist = [zindex]
    
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
        zmets = [sps.zlegend[zindex-1]]
    else:
        filterlist = None
        
    #############
    # Sum the region SFHs into a total integrated SFH, and do the
    # temporal interpolations to generate integrated spectra, LFs, and
    # SEDs
    ############
    
    # Get LFs broken out by age and metallicity as well as the total
    # LFs. these are stored as a list of different metallicities
    bins = rsed.lfbins 
    if lfstring is not None:
        lffiles = [lfstring.format(z) for z in zlist]
        lf_base = [read_lfs(f) for f in lffiles]
        lfs_zt, lf, logages = rsed.one_region_lfs(copy.deepcopy(total_sfhs), lf_base)
    else:
        lfs_zt, lf, logages = None, None, None
    # Get spectrum and magnitudes
    if filterlist is not None:
        spec, wave, mass = rsed.one_region_sed(copy.deepcopy(total_sfhs), zmets, sps )
        mags = observate.getSED(wave, spec*rsed.to_cgs, filterlist=filterlist)
        maggies = 10**(-0.4 * np.atleast_1d(mags))
    else:
        maggies, mass = None, None
    
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

    
if __name__ == '__main__':
    
    filters = ['galex_NUV', 'spitzer_irac_ch1',
               'spitzer_irac_ch4', 'spitzer_mips_24']
    
    ldir, cdir = 'lf_data/', 'angst_composite_lfs/'
    # total_cloud_data will loop over the appropriate (for the
    # isochrone) metallicities for a given lfst filename template
    lfst = '{0}z{{0:02.0f}}_tau{1:2.1f}_vega_irac{2}_n2_teffcut_lf.txt'
    basti = False
    agb_dust=1.0
    agebins = np.arange(9)*0.3 + 7.4
    
    #loop over clouds (and bands and agb_dust) to produce clfs
    for cloud in ['smc']:
        rdir = '{0}cclf_{1}_'.format(cdir, cloud)
        for band in ['2','4']:
            lfstring = lfst.format(ldir, agb_dust, band)
            dat, sfhs = total_cloud_data(cloud, filternames=filters, agb_dust=agb_dust,
                                         lfstring=lfstring, basti=basti)
            agebins = sfhs[0]['t1'][3:-1]
            outfile = lfstring.replace(ldir, rdir).replace('z{0:02.0f}_','').replace('.txt','.dat')
            write_clf_many([dat['clf_mags'], dat['agb_clf']], outfile, lfstring)
            
            fig, ax = plot_weighted_lfs(dat, agebins = agebins, dm=dmod[cloud])
            fig.suptitle('{0} @ IRAC{1}'.format(cloud.upper(), band))
            fig.savefig('byage_clfs/{0}_clfs_by_age_and_Z_irac{1}'.format(cloud, band))
            pl.close(fig)
            

            colheads = (len(agebins)-1) * ' N<m(t={})'
            colheads = colheads.format(*(agebins[:-1]+agebins[1:])/2.)
            tbin_lfs = np.array([rebin_lfs(lf, ages, agebins) for lf, ages
                                 in zip(dat['agb_clfs_zt'], dat['logages'])])
            write_clf_many([dat['clf_mags'], tbin_lfs.sum(axis=0)],
                           outfile.replace(cdir,'byage_clfs/'), lfstring,
                           colheads=colheads)
            
        pl.figure()
        for s, z in zip(sfhs, dat['zlist']):
            pl.step(s['t1'], s['sfr'], where='post', label='zind={0}'.format(z), linewidth=3)
        pl.legend(loc=0)
        pl.title(cloud.upper())

        print(cloud, dat['mstar'])
        
