import matplotlib.pyplot as pl
import astropy.io.fits as pyfits
import numpy as np

nodust = pyfits.getdata('smc.log_spitzer_irac_ch4.nodust.harris_zaritsky.fits')
dust = pyfits.getdata('smc.log_spitzer_irac_ch4.agbdust.harris_zaritsky.fits') 

cloud = 'smc'
fig, ax = pl.subplots(1,1)
image = ax.imshow(dust/nodust, interpolation = 'nearest', origin = 'lower')
ax.set_title('{0}'.format(cloud.upper()))
ax.set_xlabel('RA (pixels)')
ax.set_ylabel('Dec (pixels)')
cbar = pl.colorbar(image, orientation = 'horizontal', shrink = 0.7, pad = 0.12)
cbar.set_label(r'$F_{{8\mu m}}$(agb_dust =1)/ $F_{{8\mu m}}$(agb_dust =0)')
pl.savefig('smc.agbdust_ratio.png')
pl.close(fig)
