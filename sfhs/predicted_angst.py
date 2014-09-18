import sys, pickle, copy
import numpy as np
import matplotlib.pyplot as pl

import astropy.io.fits as pyfits
import regionsed as rsed
from lfutils import *

try:
    import fsps
    from sedpy import observate
except ImportError:
    sps =None

angst_files = {'ddo82': '10915_DDO82_F606W_F814W.zcfullfin',
               'ic2574': '9755_IC2574-SGS_F555W_F814W.zcfullfin',
               'sextansA': '',
               'ugc4305': ['10605_UGC-4305-1_F555W_F814W.zcfullfin',
                           '10605_UGC-4305-2_F555W_F814W.zcfullfin'],
               'ddo125': '',
               'ngc4163': '10915_NGC4163_F606W_F814W.zcfullfin',
               'ddo78': '10915_DDO78_F475W_F814W.zcfullfin'
               }


def total_galaxy_data(sfhfilename, zindex, filternames = None,
                      basti=False, lfstring=None, agb_dust=1.0):

    total_sfhs = [load_angst_sfh(sfhfilename)]
    rsed.lfbins += total_sfhs[0]['dmod'][0]
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
        lf_base = [read_villaume_lfs(f) for f in lffiles]
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
    total_values['clf_mags'] = bins.copy()
    total_values['logages'] = logages
    total_values['sed_ab_maggies'] = maggies
    total_values['sed_filters'] = filternames
    total_values['lffile'] = lfstring
    total_values['mstar'] = mass
    total_values['zlist'] = zlist

    rsed.lfbins -= total_sfhs[0]['dmod'][0]
    
    return total_values, total_sfhs

def load_angst_sfh(name, sfhdir = '', skiprows = 0, fix_youngest = False):
    """
    Read a `match`-produced, zcombined SFH file into a numpy
    structured array.

    :param name:
        String giving the name (and optionally the path) of the SFH
        file.
    :param skiprows:
        Number of header rows in the SFH file to skip.
    """
    #hack to calculate skiprows on the fly
    tmp = open(name, 'rb')
    while len(tmp.readline().split()) < 14:
        skiprows += 1
    tmp.close()
    ty = '<f8'
    dt = np.dtype([('t1', ty), ('t2',ty), ('dmod',ty), ('sfr',ty), ('met', ty), ('mformed',ty)])
    #fn = glob.glob("{0}*{1}*sfh".format(sfhdir,name))[0]
    fn = name
    data = np.loadtxt(fn, usecols = (0,1,2,3,6,12) ,dtype = dt, skiprows = skiprows)
    if fix_youngest:
        pass
    return data

    
if __name__ == '__main__':
    
    filters = ['galex_NUV', 'spitzer_irac_ch1',
               'spitzer_irac_ch4', 'spitzer_mips_24']
    filters = None
    ldir, cdir = 'lf_data/', 'angst_composite_lfs/'
    # total_cloud_data will loop over the appropriate (for the
    # isochrone) metallicities for a given lfst filename template
    lfst = '{0}{3}z{{0:02.0f}}_tau{1:2.1f}_vega_{2}_lf_n{4:1.0f}.txt'
    basti = False
    zindex, agb_dust, norm = 2, 1.0, 2

    adir = 'sfh_data/angst_ir_aper/'
    galaxies=['ddo78', 'ddo82']
    bands = ['f160mag', 'f814mag']
    fig, axes = pl.subplots( len(galaxies), len(bands))
    for i, band in enumerate(bands):
        for j, gal in enumerate(galaxies):
            lfstring = lfst.format(ldir, agb_dust, band, gal.replace('ddo','dd0'), norm)
            f = adir + angst_files[gal]
            outfile = '{0}_{1}_lf_n{2:1.0f}.dat'.format(gal, band, norm)
            dat, sfhs = total_galaxy_data(f, zindex, filternames=filters,
                                          agb_dust=agb_dust, lfstring=lfstring, basti=basti)
            write_clf_many([dat['clf_mags'], dat['agb_clf']], outfile, lfstring)
            
            if len(galaxies) > 1:
                ax = axes[j,i]
            elif len(bands) > 1:
                ax = axes[i]
            else: ax =axes
            ax.plot(dat['clf_mags'], dat['agb_clf'], linewidth = 3)
            ax.set_xlabel(band+' (Vega apparent)')
            ax.set_ylabel(r'$N(<m)$')
            ax.set_yscale('log')
            ax.set_xlim(24,18)
            ax.set_ylim(1e-2,1e4)
            ax.annotate(r'$N_{{tot}} = {:.0f}$'.format(np.nanmax(dat['agb_clf'])), (19.5, 5e2))
            ax.annotate(gal, (19.5, 3e3))
            #ax.set_title(gal)
    fig.suptitle('N{:1.0f}'.format(norm))
    fig.show()
    fig.tight_layout()
