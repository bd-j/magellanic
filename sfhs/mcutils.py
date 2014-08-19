import numpy as np
from hzutils import *

def xy_to_regname(x,y, cloud='smc'):
    """
    Map image pixels to region names.  If in the smc, do not give
    subregion numbers.  In the lmc, subregion numbers are always
    given.
    """
    if cloud.lower() == 'smc':
        factor = 1
    elif cloud.lower() == 'lmc':
        factor = 2
    a = ord('A')
    reg =  chr(a+x/factor), chr(a+y/factor), x % factor, y % factor
    if factor == 2:
        return '{0}{1}_{2}{3}'.format(*reg)
    else:
        return '{0}{1}'.format(*reg[0:2])

def regname_to_xy(regname, cloud='smc'):
    """
    Map region names to image pixels, accounting for the coadded
    pixels in the LMC regions.
    """
    n = regname
    a = ord('A')
    x, y = ord(n[0]) - a, ord(n[1]) - a 
    if cloud.lower() == 'smc':
        #done
        return (x,) , (y,)
    elif cloud.lower() == 'lmc':
        if len(n) > 2:
            #there is a subgrid designation
            sx, sy = int(n[3]), int(n[4])
            return (x*2 + sx,) , (y*2 + sy,)
        else:
            # there is no subgrid designation and we return the x,y of
            # all four subgrid pixels
            x = x*2 + np.array([0,0,1,1])
            y = y*2 + np.array([0,1,0,1])
            return tuple(x), tuple(y)
        
def mc_ast(cloud):
    """
    Set up the sky coordinate system.
    """
    if cloud.lower() == 'lmc':
        #regions = utils.lmc_regions()
        nx, ny, dm = 48, 38, 18.5
        cdelt = [24./60./np.cos(np.deg2rad(-69.38333)),  24./60.]
        crpix = [ 0, 0]
        crval = [ 67.75 - cdelt[0]/2, -72.3833]
    elif cloud.lower() == 'smc':
        #regions = utils.smc_regions()
        nx, ny, dm = 20, 23, 18.9
        #the minimum ra in deg of the 'A' corner
        crval = [6.25, -74.95]
        crpix = [0., 0.]
        #the dra in deg worked out from figure 3 of H & Z
        cdelt = [0.5/11 * 15., 12./60.]
    return crpix, crval, cdelt, [nx, ny]#, regions


def parse_locstring(locstring):
    loc = locstring.split()    
    ddeg = float(loc[3][:-1])
    dec = np.sign(ddeg) * (abs(ddeg) + float(loc[4][:-1])/60.) #use reported coords
    ra = 15.*( float(loc[1][:-1]) + float(loc[2][:-1])/60.)
    return ra, dec


