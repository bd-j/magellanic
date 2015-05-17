import matplotlib.pyplot as pl
import numpy as np
import pickle
import astropy.io.fits as pyfits
from lfutils import *

smccols = {'RA': 'RAJ2000',
           'DEC': 'DEJ2000',
           'IRAC4': '__8_0_',
           'IRAC2': '__4_5_',
           'IRAC4_err': 'e__8_0_',
           'IRAC2_err': 'e__4_5_',
           'STARTYPE': 'Class',
           'agb_codes': ['C-AGB', 'O-AGB', 'x-AGB', 'aO-AGB'],#'FIR'],
           'corners': ''
          }
lmccols = {'RA': 'RA',
           'DEC': 'DEC',
           'IRAC4': 'IRAC4',
           'IRAC2': 'IRAC2',
           'IRAC4_err': 'DIRAC4',
           'IRAC2_err': 'DIRAC2',
           'STARTYPE': 'FLAG',
           'agb_codes': ['C', 'O', 'X', 'aO/C', 'aO/O'],# 'RS-C', 'RS-O', 'RS-X', 'RS-a'],
           'corners': ''
           }

#rdir = '/Users/bjohnson/Projects/magellanic/sfhs/results_predicted/'

def photometer(imname, region):
    import astropy.wcs as pywcs
    image = pyfits.getdata(imname)
    header = pyfits.getheader(imname)
    wcs = pywcs.WCS(header)
    ps = np.hypot(*wcs.wcs.cd*3600.)
    yy, xx = np.indices(image.shape)
    ra, dec = wcs.wcs_pix2world(xx.flatten(), yy.flatten(), 0)
    sel = region.contains(ra.flatten(), dec.flatten())
    flux = np.nansum(image.flatten()[sel])
    abmag = -2.5*np.log10((flux * 1e6*2.35e-11*ps.prod())/3631.)
    return abmag

def select(catalog, coldict, region=None, codes=None, **extras):
    """Select stars of types given by ``codes`` from the supplied ``catalog``.

    :param catalog:
        Nd structured array.

    :param coldict:
        A dictionary giving a mapping from 'RA', 'DEC', and 'STARTYPE'
        to the corresponding field names in the supplied catalog.

    :param region:
        If supplied, a sedpy.ds9region.Region object which has the
        method ``contains`` defined.  Only catalog objects contained
        in the region are returned

    :param codes:
        A list of string types giving the 'STARTYPE' codes to select

    :returns subcat:
        The subset of the supplied catalog that is of type ``codes``
        and within the supplied region.
    """
    x, y = catalog[coldict['RA']], catalog[coldict['DEC']]
    if region is None:
        sel = np.ones(len(catalog), dtype=bool)
    else:
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


def bounding_hull(catalog, coldict, **extras):
    """Compute the convex hull for a catalog and return a string of
    the coordinates of the vertices, as well as a two element list of
    arrays of the RAs and Decs of the vertices.
    """
    from scipy.spatial import ConvexHull
    points = np.array([catalog[coldict[f]] for f in ['RA','DEC']]).T
    hull = ConvexHull(points)
    v =  points[hull.vertices,0], points[hull.vertices,1]
    #v = [ np.concatenate((v[i], np.array([v[i][0]]))) for i in [0,1] ]
    
    corners = ','.join([str(val) for pair in zip(v[0], v[1]) for val in pair])
    return corners, v

def mcps_corners(cloud):
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
    """Shortcut method to give the catalog and coldict mapping for a
    given cloud.
    """   
    c = cloud.lower()
    catname = '/Users/bjohnson/Projects/magellanic/catalog/boyer11_{}.fits.gz'.format(c)
    catalog = pyfits.getdata(catname)
    if c == 'smc':
        cols = smccols
    elif c == 'lmc':
        cols = lmccols
    return catalog, cols

if __name__ == '__main__':
    from sedpy import ds9region as ds9reg
    clouds, agb_dust = ['smc', 'lmc'], 1.0
    for cloud in clouds:

        bands = ['IRAC2', 'IRAC4']
        # Get the observed CLFs
        defstring = mcps_corners(cloud)
        region = ds9reg.Polygon(defstring)
        cat, cols = cloud_cat(cloud)
        subcat = select(cat, cols, region, codes=cols['agb_codes'])
        for band in bands:
            obs_clf = cumulative_lf(subcat, cols[band])
            fstring = 'composite_lfs/obs_clf.{0}.{1}'
            write_clf(obs_clf, fstring.format(*values[0:2])+'.dat', 'Observed')


