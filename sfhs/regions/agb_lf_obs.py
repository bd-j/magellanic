import numpy as np
import astropy.io.fits as pyfits

catname = '/Users/bjohnson/Projects/magellanic/catalog/boyer11.fits'
rafield, decfield = 'RAJ200', 'DEJ2000'

def main():
    catalog = pyfits.getdata(catname)
    regions = utils.smc_regions()
    
    for name, dat in regions.iteritems():
        defstring = corners_of_region(name, dat['loc'], cloud, string=True)
        ds9reg = ds9.Polygon(defstring)
        this = select(catalog, ds9reg)
        

def select(catalog, region):
    sel = region.contains(catalog[rafield], catalog[defield])
    sel = sel & (catalog['Class'] != 'RSG') & (catalog['Class'] != 'RGB')
    return sel

def corners_of_region(regname, regloc, cloud, string=False):
    """
    Use a defnition of the grid worked out from figure 3 of H & Z
    2004.  The idea is that the given central coordinates are not
    precise enough, but that the finest grid size is constant in Ra
    and Dec
    """
    #get the astrometry
    crpix, crval, cd, [nx, ny] = regutils.mc_ast(cloud)
    rcorners = np.array([0,0,1,1]) * cd[0]
    dcorners = np.array([0,1,1,0]) * cd[1]
    #get the pixel values
    x, y = regutils.regname_to_xy(regname, cloud)
    ra = (x-crpix[0])*cd[0] + crval[0]
    dec = (y-crpix[1])*cd[1] + crval[1]
    sz = np.size(ra)
    if sz > 1:
        rc = ra[0] + rcorners * np.sqrt(sz)
        dc = dec[0] + dcorners * np.sqrt(sz)
    elif sz == 1:
        rc = ra + rcorners
        dc = dec + dcorners

    if string:
        pass
        
    
def hz_corners_of_region(regname, regions, cloud):
    """
    Try to generate regions using the Harris and Zaritsky scheme.
    From Zaritsky:

    "If you take the midpoint between adjacent boxes as the boundaries
    (the coordinates as given are box centers), you'll be matching
    what we did."

    However, it's not clear what to do in the cases of edges and this
    doesn't really make sense for the variable box sizes of the LMC,
    and results in variable region shapes due to roundoff errors in
    the central coordinates.
    """

if __name__ == '__main__':
    main()
 

