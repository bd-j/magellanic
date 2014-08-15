import numpy as np
import aplpy
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
from astropy.io import ascii
import astropy.time as time
import utils

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
        alpha  = np.clip(d[etfield.keys()[0]]/etfield.values()[0], 0.1, 0.3)
        circle = Circle((x,y), radius/f[0].header['CD1_1'],
                        alpha=alpha, **extra)
        pl.gca().add_artist(circle)

tilelist = '/Users/bjohnson/SFR_FIELDS/Nearby/MCs/GR67.txt'
cloud = 'LMC'

if cloud is 'LMC':
    lmcdir = '/Users/bjohnson/SFR_FIELDS/Nearby/MCs/LMC/'
    ffname = lmcdir + 'lmc_bothun_R_ast.fits'
    h2d = 1
    vmin, vmax = 0, 0.5e4
    pixn = np.arange(0,900, 100)
elif cloud is 'SMC':
    smcdir = '/Users/bjohnson/SFR_FIELDS/Nearby/MCs/SMC/'
    ffname = smcdir + 'smc_bothun_R_ast.fits'
    h2d = 1
    vmin, vmax = 1000, 5000
    pixn = np.arange(0,900, 100)

gdata = utils.read_galex_tilelist(tilelist)

f = pyfits.open(ffname)
WCS = pywcs.WCS(f[0].header)

near = ( (np.abs(gdata['ra_cent'] - f[0].header['CRVAL1']) < 16) &
         (np.abs(gdata['dec_cent'] - f[0].header['CRVAL2']) < 16)
         )

gdata = gdata[near]

#galex = galex_read_tiles_csv(tilelist_csv)

radius = 0.55

release = ['tile_gr6', 'tile_gr7_new', 'tile']
rtitle  = ['GR6', 'new in GR7 (feb2013)', 'GR6+7']

for i,rel in enumerate(release):
    print(rel)
    pl.figure()
    pl.imshow(f[0].data, vmin = vmin, vmax = vmax, origin = 'lower',
          interpolation = 'nearest', cmap = 'gray')
    
    thisfuv = ((np.char.find(gdata['release'],rel) >=0) & 
               (gdata['fuv_exptime'] > 0)
               )
    plot_circles(gdata[thisfuv], WCS, etfield = {'fuv_exptime':1000}, facecolor = 'c')
    thisnuv = ((np.char.find(gdata['release'],rel) >=0) & 
               (gdata['nuv_exptime'] > 0)
               )
    plot_circles(gdata[thisnuv], WCS, facecolor = 'y')
    
    pl.title(rtitle[i])
    pl.xlabel('RA')
    pl.ylabel('Dec')
    ra_arr = (WCS.wcs_pix2world(pixn,np.zeros_like(pixn),1))[0]
    ralist = ['{0:3.2f}'.format(r) for r in ra_arr]
    dec_arr = (WCS.wcs_pix2world(np.zeros_like(pixn),pixn,1))[1]
    declist = ['{0:3.2f}'.format(r) for r in dec_arr]
    pl.xticks(pixn, ralist)
    pl.yticks(pixn, declist)
    pl.show()
    pl.savefig('{0}_{1}.png'.format(cloud,rel))
    
    #pl.figure()
    #pl.hist(gdata[thisnuv]['nuv_exptime'], range = (0, 2000))
    #pl.title(rtitle[i])


#APLpy version -- won't respect alpha and facecolor at the same time
gc = aplpy.FITSFigure(f)
gc.show_grayscale()


#raise ValueError()

#gc.show_circles(5.4*15., -68.0, 0.55, alpha = 0.3, edgecolor = 'y', layer = 'tile')
#gc.save('test.png',transparent = True)
#gc.show_circles(galex['RA_cent'][inds], galex['Dec_cent'][inds], radius[inds], color = 'y', alpha = 0.1)

  
