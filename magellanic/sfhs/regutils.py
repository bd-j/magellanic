import numpy as np
import mcutils


def corners_of_region(regname, cloud, string=False):
    """
    Use a defnition of the grid worked out from figure 3 of H & Z
    2004.  The idea is that the given central coordinates are not
    precise enough, but that the finest grid size is constant in Ra
    and Dec
    """
    #get the astrometry
    crpix, crval, cd, [nx, ny] = mcutils.mc_ast(cloud)
    rcorners = np.array([0,0,1,1]) * cd[0]
    dcorners = np.array([0,1,1,0]) * cd[1]
    #get the pixel values
    x, y = mcutils.regname_to_xy(regname, cloud)
    x, y = np.array(x), np.array(y)
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
        tmp = ','.join([ str(val) for pair in zip(rc, dc) for val in pair])
        return 'polygon', tmp
    return 'polygon', rc, dc



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
    raise(NotImplementedError)
