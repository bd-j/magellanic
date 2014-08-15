import numpy as np
import aplpy
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
from astropy.io import ascii
import astropy.time as time

import matplotlib.pyplot as pl
from matplotlib.patches import Circle
#from matplotlib.collections import PatchCollection

def plot_circles(data, wcs, etfield = {'nuv_exptime':1000}, radius = 0.55, **extra):
    for d in data:
        #print(d['ra_cent'],d['dec_cent'])
        x,y = wcs.wcs_world2pix(d['ra_cent'],d['dec_cent'], 1)
        #alpha = 0.2
        #if d[etfield.keys()[0]] > etfield.values()[0]:
        #    alpha = 0.6
        alpha  = np.clip(d[etfield.keys()[0]]/etfield.values()[0], 0.3, 0.5)
        circle = Circle((x,y), radius/f[0].header['CDELT1'],
                        alpha=alpha, **extra)
        pl.gca().add_artist(circle)


cloud = 'LMC'

if cloud is 'LMC':
    lmcdir = '/Users/bjohnson/SFR_FIELDS/Nearby/MCs/LMC/'
    ffname = lmcdir + 'lmc_parker_r_ast_ncp.fits'
    #ffname = lmcdir + 'lmc_iras60_nc_ncp.fits'
    tilelist_csv = lmcdir + 'LMC_galextiles_table.csv'
    h2d = 15
    vmin, vmax = 0, 0.5e4
elif cloud is 'SMC':
    smcdir = '/Users/bjohnson/SFR_FIELDS/Nearby/MCs/SMC/'
    ffname = smcdir + 'smc_bothun_R_ast.fits'
    ffname = smcdir + 'smc_iras60_tan_ncp.fits'
    tilelist_csv = smcdir + 'SMC_galextiles_table.csv'
    h2d = 1

gdata = ascii.read(tilelist_csv+'.nocomma', fill_values = [('---','-999'), ('--','-99')])

f = pyfits.open(ffname)
f[0].header['CRVAL1'] *= h2d
f[0].header['CDELT1'] *= h2d
WCS = pywcs.WCS(f[0].header)

#galex = galex_read_tiles_csv(tilelist_csv)

radius = 0.55

date_lim =[54750, 55500, 56000]
time_lim = time.Time(date_lim, format = 'mjd', scale = 'utc')
time_lim.out_subfmt = 'date'

for i,d in enumerate(date_lim):
    pl.figure()
    pl.imshow(np.clip(f[0].data, vmin, vmax),
          interpolation = 'nearest', cmap = 'gray')
    
    thisfuv = ((gdata['meanObsMJD'] < d) &
               (gdata['fuv_exptime'] > 0) & 
               (gdata['spectra'] == 'false')
               )
    plot_circles(gdata[thisfuv], WCS, etfield = {'fuv_exptime':1000}, facecolor = 'b')
    thisnuv = ((gdata['meanObsMJD'] < d) &
               (gdata['nuv_exptime'] > 0) & 
               (gdata['spectra'] == 'false')
               )
    plot_circles(gdata[thisnuv], WCS, facecolor = 'y')
    spectra = ((gdata['meanObsMJD'] < d) &
               (gdata['spectra'] == 'true')
               )
    plot_circles(gdata[spectra], WCS, etfield = {'nSpectra':600},facecolor = 'c')
    pl.title('Obs date < {0}'.format(time_lim.iso[i]))
    pl.xlabel('RA')
    pl.ylabel('Dec')
    #pl.xticks(xpix, WCS.wcs_pix2world(xpix,np.zeros_like(xpix),1))
    pl.show()


#APLpy version -- won't respect alpha and facecolor at the same time
gc = aplpy.FITSFigure(f)
gc.show_grayscale()
pl.show()

raise ValueError()

gc.show_circles(5.4*15., -68.0, 0.55, alpha = 0.3, edgecolor = 'y', layer = 'tile')
gc.save('test.png',transparent = True)
#gc.show_circles(galex['RA_cent'][inds], galex['Dec_cent'][inds], radius[inds], color = 'y', alpha = 0.1)

  
def read_galex_tilelist(filename):
    f = open(filename,'r')
    lines = f.readlines()
    f.close()
    nline = len(lines)
    bnum = 0
    
    ttype = ['header,''cal_new', 'spec_gr7_new','spec_gr6_pluset','tile_gr7_new',
             'tile_gr7_new_css', 'tile_gr6_pluset','gr6_css','gr6']

    dt = {'names':('survey','tilename','tilenum',
                   'subgrid','ra_cent','dec_cent',
                   'nuv_exptime','fuv_exptime','release'),
          'formats':('a3','a32','a5','a4','<f8','<f8','<f8','<f8','a10')}
    galex = np.zeros(nline,dtype = dt)
    
    for i,l in enumerate(lines):
        s = l.split()
        if s[0] in ['CAI','AIS','MIS','DIS','GII','NGS']:
            galex[i] = (s[0], s[1], s[2], s[3], float(s[4]), float(s[5]),
                    float(s[6]),float(s[7]), ttype[bnum])
        elif s[0] is 'Listing':
            bnum += 1
            print(i,s[:7],ttype[bnum])

    return galex



