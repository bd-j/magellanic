import os, copy
import numpy as np

mcdir, f = os.path.split(__file__)
lmcfile = os.path.join(mcdir,'sfh_data','lmc_sfh.dat')
smcfile = os.path.join(mcdir,'sfh_data','smc_sfh.dat')

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

def regname_to_xy(regname, cloud='smc', **extras):
    """
    Map region names to image pixels, accounting for the coadded
    pixels in the LMC regions.
    """
    n = copy.deepcopy(regname)
    a = ord('A')
    x, y = ord(n[0]) - a, ord(n[1]) - a 
    if cloud.lower() == 'smc':
        #done
        return [x] , [y]
    elif cloud.lower() == 'lmc':
        if len(n) > 2:
            #there is a subgrid designation
            sx, sy = int(n[3]), int(n[4])
            return [x*2 + sx] , [y*2 + sy]
        else:
            # there is no subgrid designation and we return the x,y of
            # all four subgrid pixels
            x = x*2 + np.array([0,0,1,1])
            y = y*2 + np.array([0,1,0,1])
            return list(x), list(y)
        
def mc_ast(cloud, badenes=True, **extras):
    """
    Set up the sky coordinate system.
    """
    if cloud.lower() == 'lmc':
        #regions = utils.lmc_regions()
        nx, ny, dm = 48, 38, 18.5
        if badenes:
            # Use the Badenes reverse-engineering of the coordinates
            cdelt = [0.5625,  0.2]
            crval = [ 67.3125, -72.5]
            crpix = [0.0, 0.0]
        else:
            # Use my reverse engineering of the coordinates
            cdelt = [24./60./np.cos(np.deg2rad(-69.38333))/2.,  24./60./2.]
            crval = [ 67.75 - cdelt[0]/2, -72.3833]
            crpix = [ 0.5, 0.5]
        
    elif cloud.lower() == 'smc':
        #regions = utils.smc_regions()
        nx, ny, dm = 20, 23, 18.9
        #the minimum ra in deg of the 'A' corner
        crval = [6.25, -74.95]
        crpix = [0.5, 0.5]
        #the dra in deg worked out from figure 3 of H & Z
        cdelt = [0.5/11 * 15., 12./60.]
    return crpix, crval, cdelt, [nx, ny]#, regions

def corners_of_region(regname, cloud, string=False, **kwargs):
    """
    Use a defnition of the grid worked out from figure 3 of H & Z
    2004.  The idea is that the given central coordinates are not
    precise enough, but that the finest grid size is constant in Ra
    and Dec
    """
    #get the astrometry
    crpix, crval, cd, [nx, ny] = mc_ast(cloud, **kwargs)
    rcorners = np.array([0,0,1,1]) * cd[0]
    dcorners = np.array([0,1,1,0]) * cd[1]
    #get the pixel indices
    x, y = regname_to_xy(regname, cloud)
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

def parse_locstring(locstring):
    loc = locstring.split()    
    ddeg = float(loc[3][:-1])
    dec = np.sign(ddeg) * (abs(ddeg) + float(loc[4][:-1])/60.) #use reported coords
    ra = 15.*( float(loc[1][:-1]) + float(loc[2][:-1])/60.)
    return ra, dec

def make_ds9(cloud, out=None):
    """Make a ds9 region file containing all of the regions
    """
    from sedpy import ds9region as ds9
    if out is None:
        out = '{0}_regions_grid.reg'.format(cloud)
    out = open(out,'w')
    if cloud.lower() == 'lmc':
        regions = mcutils.lmc_regions()
    else:
        regions = mcutils.smc_regions()
    regions.pop('header')
    for name, dat in regions.iteritems():
        shape, defstring = corners_of_region(name, cloud, string=True)
        ds9reg = ds9.Polygon(defstring)
        ds9reg.print_to(fileobj=out, color='red', label=name)
    out.close()

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

def mc_regions(cloud):
    if cloud.lower() == 'smc':
        reg = smc_regions()
    elif cloud.lower() == 'lmc':
        reg = lmc_regions()
    return reg
    
def lmc_regions(filename=lmcfile):
    """Read the Harris and Zaritsky LMC data file ( obtained from
    http://djuma.as.arizona.edu/~dennis/mcsurvey/Data_Products.html )
    into a dictionary of SFHs.

    :param filename:
        The path to the data file.
    
    :returns region_dict:
        A dictionary of LMC regions. Each key of the returned
        dictionary is a region name, and each value is a dictionary
        containing a list of SFHs for that region (one for each
        metallicity), a list of metallicities of each SFH, and a
        location string giving the RA, Dec coordinates of the region.
        There is also one key in the dictionary, 'header', containing
        header information from the original files.  Each of the SFHs
        is a structured ndarray that can be input to scombine methods.
    """

    regions = {}
    k = 'header'
    regions[k] = []
    
    f = open(filename, 'rb')
    for line in f:
        line = line.strip()
        if line.find('Region') >=0:
            k = line.split()[-1]
            regions[k] = {'sfhs':[]}
        elif line.find('(') == 0:
            regions[k]['loc'] = line
        elif len(line.split()) > 0 and line[0] != '-':
            if k == 'header':
                regions[k] += [line]
            else:
                regions[k]['sfhs'] += [[float(c) for c in line.split()]]
    f.close()
    for k, v in regions.iteritems():
        if k == 'header':
            continue
        sfhs, zs = process_lmc_sfh( v['sfhs'])
        regions[k]['sfhs'] = sfhs
        regions[k]['zmet'] = zs
        
    return regions

def process_lmc_sfh(dat):
    """
    Take a list of lists, where each row is a time bin, and convert it
    into several SFHs at different metallicities.
    """
    all_sfhs = []
    zlegend = np.array([0.001, 0.0025, 0.004, 0.008])
    usecol = [10,7,4,1]
    
    s = np.array(dat)
    inds = np.argsort(s[:,0])
    s = s[inds, :]
    wt = np.diff(s[:,0]).tolist()
    wt = np.array([wt[0]]+wt)
    nt = s.shape[0]
    ty = '<f8'
    dt = np.dtype([('t1', ty), ('t2',ty), ('dmod',ty), ('sfr',ty), ('met', ty), ('mformed',ty)])
    for zindex, zmet in enumerate(zlegend):
        data = np.zeros(nt, dtype = dt)
        data['t1'] = s[:,0] - wt
        data['t2'] = s[:,0]
        data['met'] = np.log10(zmet/0.019)
        data['sfr'] = s[:, usecol[zindex]] * 1e-6
        data['dmod'] = 18.50
        all_sfhs += [data]

    return all_sfhs, zlegend
    
def smc_regions(filename=smcfile):
    """Read the Harris and Zaritsky SMC data file ( obtained from
    http://djuma.as.arizona.edu/~dennis/mcsurvey/Data_Products.html )
    into a dictionary of SFHs.

    :param filename:
        The path to the data file.
    
    :returns region_dict:
        A dictionary of SMC regions. Each key of the returned
        dictionary is a region name, and each value is a dictionary
        containing a list of SFHs for that region (one for each
        metallicity), a list of metallicities of each SFH, and a
        location string giving the RA, Dec coordinates of the region.
        There is also one key in the dictionary, 'header', containing
        header information from the original files.  Each of the SFHs
        is a structured ndarray that can be input to scombine methods.
    """
    #wow.  really?  could this *be* harder to parse?
    #why so differnt than lmc?
    regions ={'header':[]}
    f = open(filename, 'rb')
    for i, line in enumerate(f):
        line = line.strip()
        if i < 26:
            regions['header'] += [line]
        else:
            cols = line.split()
            reg = cols[0]
            if len(cols)  == 13:
                #padding.  because who does this?  unequal line lengths?
                cols = cols + ['0','0','0']
            
            regions[reg] = regions.get(reg, []) + [[float(c) for c in cols[1:]]]
    f.close()
    #reprocess to be like lmc
    for k,v in regions.iteritems():
        if k == 'header':
            continue
        sfhs, zs, loc = process_smc_sfh( v )
        regions[k] = {}
        regions[k]['sfhs'] = sfhs
        regions[k]['zmet'] = zs
        regions[k]['loc'] = loc
        
    return regions

def process_smc_sfh(dat):
    """
    Take a list of lists, where each row is a time bin, and convert it
    into several SFHs at different metallicities.
    """
    all_sfhs = []
    zlegend = np.array([0.001, 0.004, 0.008])
    usecol = [12,9,6]
    
    s = np.array(dat)
    inds = np.argsort(s[:,4])
    s = s[inds, :]
    nt = s.shape[0]
    ty = '<f8'
    dt = np.dtype([('t1', ty), ('t2',ty), ('dmod',ty), ('sfr',ty), ('met', ty), ('mformed',ty)])
    for zindex, zmet in enumerate(zlegend):
        data = np.zeros(nt, dtype = dt)
        data['t1'] = s[:,4]
        data['t2'] = s[:,5]
        data['met'] = np.log10(zmet/0.019)
        data['sfr'] = s[:, usecol[zindex]] * 1e-6
        data['dmod'] = 18.50
        all_sfhs += [data]
    loc = "( {0:02.0f}h {1:02.0f}m {2:02.0f}d {3:02.0f}m )".format(*dat[0][0:4])
    return all_sfhs, zlegend, loc
