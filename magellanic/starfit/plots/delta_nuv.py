####
#1) Generate a predicted UV image by summing the predicted UV flux of stars in a given pixel
#2) Generate an image of < (NUV_obs - NUV_pred)/A_V > for largish regions, averaged over stars with NUV_pred between 16 and 18th mag

import numpy as np
import astropy.io.fits as pyfits
import astropy.wcs as wcs




f = pyfits.open(image_name)
nuv_image = f[0].data
nuv_header = f[0].header
WCS = wcs.WCS(nuv_header)
f.close()

f = pyfits.open(catalog_name)
cat = f[1].data
f.close()

x, y = (WCS.wcs_world2pix(cat['RAh']*15., cat['Dec'], 0)).astype(int)

nuv_pred = np.zeros_like(nuv_image)
nuv_pred[(x,y)] += 10**((20.08-cat['NUV_p500'])*0.4))

