import pyfits
import numpy as np


catname = '../results/mcps_lmc_test_starprops.fits'
catname = '../results/mcps_lmc_V20_detect4_starprops.fits'
f = pyfits.open(catname)

mcps = f[1].data
g = mcps['NUV_p500'] < 19

out = open('mcps_coords.reg','wb')
for d in mcps[g] : out.write('{0}  {1}\n'.format(d['RAh']*15, d['Dec']))
out.close()

