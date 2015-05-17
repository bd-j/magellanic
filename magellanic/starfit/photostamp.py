import numpy as np
import os, time
import pyfits
from astropy import wcs
import matplotlib.pyplot as pl
import hrdspyutils as utils
import photometer
from numpy.lib.recfunctions import merge_arrays


## User parameters
rp = {}
rp['stampcatname'] = 'results/massey_lmc_test_starprops_stamped.fits'
rp['stampcatname'] = 'results/mcps_lmc_V20_detect4_starprops_stamped.fits'
rp['label'] = 'MCPS'
rp['radius'] = 3
rp['subpixels'] = 2
rp['inner'], rp['outer'] = 11,13
rp['bgtype'] = 'quartile_sky' #quartile_sky | gaussfit_sky | sigclip_sky

## Initialize aperture (and background annulus)
ap = photometer.Circular()
#ap.background = photometer.ZeroSky(return_value = (0,1,0))
ap.background = photometer.Annulus(bgtype = rp['bgtype'])

## Read the catalog, and plot average stamp
f = pyfits.open(rp['stampcatname'])
scat = f[1].data
WCS = wcs.WCS(f[1].header)
f.close()
x, y = WCS.wcs_world2pix(scat['RAh']*15, scat['Dec'], 0)

avg = scat['nuv_stamp'].mean(0)
pl.figure()
pl.imshow(avg, interpolation = 'nearest', origin = 'lower')

## Build aperture dictionaries
nuv_mag = np.zeros(len(scat))
stampsize = scat[0]['nuv_stamp'].shape[0]
shape = {'xcen': stampsize/2.,
         'ycen': stampsize/2.,
         'radius': rp['radius'],
         'subpixels': rp['subpixels']}
skypars = {'xcen': stampsize/2.,
           'ycen': stampsize/2.,
           'radius': rp['outer'], 'inner_radius':rp['inner'],
           'subpixels': 1,
           'sigma': [5.0,3.0]}

## Loop over catalog entries
for i,star in enumerate(scat):

    shape['xcen'] = x[i] - star['cx_stamp']+stampsize/2.
    shape['ycen'] = y[i] - star['cy_stamp']+stampsize/2.
    skypars['xcen'] = shape['xcen']
    skypars['ycen'] = shape['ycen']

    nuv_flux, junk = ap.measure_flux(shape, star['nuv_stamp'], skypars = skypars)
    nuv_mag[i] = 20.08 - 2.5*np.log10(nuv_flux)

pcat = utils.join_struct_arrays( [scat, utils.structure_array(nuv_mag, ['galex_NUV'])] )

raise ValueError('wait')


pl.figure()
pl.scatter(scat['NUV_p500'], nuv_mag-scat['NUV_p500'],
           alpha = 0.3, color = 'b', label = rp['label'])
pl.ylabel(r'$\Delta$NUV (obs-pred)')
pl.xlabel('NUV (p50)')
pl.show()

#pl.scatter(scat['NUV_p500'], nuv_mag-scat['NUV_p500'],
#           alpha = 0.3, color = 'r', label = rp['label'])

gg = (scat['NUV_p500'] > 15) & (scat['NUV_p500'] < 18)
pl.figure()
pl.hist((np.abs(nuv_mag-scat['NUV_p500'] - 3)/(scat['NUV_p975']-scat['NUV_p025']))[gg], range = (0,5), bins = 30)
pl.figure()
pl.hist((np.abs(nuv_mag-scat['NUV_p500'] - 3)/(scat['NUV_p975']-scat['NUV_p025'])/2.), range = (-5,5), bins = 30)
