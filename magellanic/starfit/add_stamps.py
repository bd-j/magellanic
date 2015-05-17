import numpy as np
import os, time
#import astropy.io.fits as pyfits
import pyfits
from astropy import wcs
import hrdspyutils as utils

rp={}
rp['stamp_size'] = 31 #in pixels, odd number
rp['catname'] = 'results/massey_lmc_test_starprops.fits'
#rp['catname'] = 'results/mcps_lmc_V20_detect4_starprops.fits'
rp['imname'] = '/Users/bjohnson/DATA/magellanic/LMC/images/lmc_int_corr_6.0_v3.fits'

f = pyfits.open(rp['catname'])
cat = f[1].data
f.close()
g = cat['NUV_p500'] < 20
cat = cat[g]

f = pyfits.open(rp['imname'])
im = f[0].data
imhdr = f[0].header
WCS = wcs.WCS(f[0].header)
f.close()
copypar = ['CTYPE1','CTYPE2','CDELT1','CDELT2', 'EQUINOX',
           'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CROTA2']

dt = {'names': ('nuv_stamp', 'cx_stamp', 'cy_stamp'),
      'formats': (('<f8',(rp['stamp_size'],rp['stamp_size'])), '<i8', '<i8')
      }
imrec = np.zeros(len(cat), dtype = dt)

cx, cy = np.round(WCS.wcs_world2pix(cat['RAh']*15.,cat['Dec'], 0)).astype('<i8')
imrec['cx_stamp'], imrec['cy_stamp'] = cx, cy

sx = rp['stamp_size']
sy = rp['stamp_size']
cols = np.arange(sx) - (sx-1)/2
rows = np.arange(sy) - (sy-1)/2
#2D array of 1-d indices of a subarray
patch = (cols[:,None]*im.shape[1] + rows[None,:] )
#3D array of subarrays for each center
inds = patch[None,...] + (cy*im.shape[1]+cx)[:,None,None]
inds = inds.transpose(0,2,1)
#use the 1-d subscripts to subscript the raveled image
imrec['nuv_stamp'] = im.ravel()[inds]
inds = 0

print(imrec.dtype)

cat = utils.join_struct_arrays([cat,imrec])

print(cat.dtype)
print(imrec.dtype)

cols = pyfits.ColDefs(cat)
tbhdu = pyfits.new_table(cols)
for p in copypar: tbhdu.header[p] = imhdr[p]

tbhdu.writeto('{0}_stamped.fits'.format(rp['catname'][:-5]), clobber = True)

