import matplotlib.pyplot as pl
import numpy as np
import pickle
import astropy.io.fits as pyfits
from sedpy import ds9region as ds9reg
from lfutils import *

smccols = {'RA': 'RAJ2000',
           'DEC': 'DEJ2000',
           'IRAC4': '__8_0_',
           'IRAC2': '__4_5_',
           'IRAC4_err': 'e__8_0_',
           'IRAC2_err': 'e__4_5_',
           'STARTYPE': 'Class',
           'agb_codes': ['C-AGB', 'O-AGB', 'x-AGB', 'aO-AGB','FIR']
          }
lmccols = {'RA': 'RA',
           'DEC': 'DEC',
           'IRAC4': 'IRAC4',
           'IRAC2': 'IRAC2',
           'IRAC4_err': 'DIRAC4',
           'IRAC2_err': 'DIRAC2',
           'STARTYPE': 'FLAG',
           'agb_codes': ['C', 'O', 'X', 'aO/C', 'aO/O', 'RS-C', 'RS-O', 'RS-X', 'RS-a']
           }

rdir = '/Users/bjohnson/Projects/magellanic/sfhs/results_predicted/'


def select(catalog, coldict, region, codes=None):
    x, y = catalog[coldict['RA']], catalog[coldict['DEC']]
    sel = region.contains(x, y)
    if codes is not None:
        typesel = False
        for c in codes:
            typesel |= (catalog[coldict['STARTYPE']] == c)
        sel = sel & typesel
    return catalog[sel]

    
def cumulative_obs_lf(catalog, bandname):
    mag = catalog[bandname]
    mag = mag[np.isfinite(mag)]
    order = np.argsort(mag)
    return mag[order], np.arange(len(mag))


def cloud_corners(cloud):
    """
    Return strings defining vertices of the polygon enclosing the MCPS
    survey.  These are very approximate.
    """
    c =cloud.lower()
    if c == 'smc':
        corners = '6.25,-74.95,19.0,-74.95,19.0,-70.533,6.25,-70.533'
    elif c == 'lmc':
        corners = '70.0,-72.2,92,-72.2,90,-65.4,72.5,-65.4'
    return corners

    
def cloud_cat(cloud):
       
    c = cloud.lower()
    catname = '/Users/bjohnson/Projects/magellanic/catalog/boyer11_{}.fits.gz'.format(c)
    catalog = pyfits.getdata(catname)
    if c == 'smc':
        cols = smccols
    elif c == 'lmc':
        cols = lmccols
    return catalog, cols

if __name__ == '__main__':
    clouds, agb_dust = ['smc', 'lmc'], 1.0
    for cloud in clouds:

        bands = ['IRAC2', 'IRAC4']
        # Get the observed CLFs
        defstring = cloud_corners(cloud)
        region = ds9reg.Polygon(defstring)
        cat, cols = cloud_cat(cloud)
        subcat = select(cat, cols, region, codes=cols['agb_codes'])
        for band in bands:
            obs_clf = cumulative_lf(subcat, cols[band])
            fstring = 'results_compare/obs_clf.{0}.{1}'
            write_clf(obs_clf, fstring.format(*values[0:2])+'.dat', 'Observed')


