import numpy as np
import sfhutils as utils

cloud = 'lmc'
if cloud.lower() == 'lmc':
    regions = utils.lmc_regions()
    out = open('lmc_regions{}.reg'.format(hzext) ,'w')
    dm = 18.5
elif cloud.lower() == 'smc':
    regions = utils.smc_regions()
    out = open('smc_regions{}.reg'.format(hzext) ,'w')
    dm = 18.9

corners = np.array([1,0,0,1]), np.array([0,0,1,1])
region_center = np.zeros()

for k, v in regions.iteritems():
    x =
    y =

    loc = v['loc'].split()    
    ddeg = float(loc[3][:-1])
    dec = np.sign(ddeg) * (abs(ddeg) + float(loc[4][:-1])/60.) #use reported coords
    ra = 15.*( float(loc[1][:-1]) + float(loc[2][:-1])/60.)

for i in range():
    for j in range():
        ra = cra[i,j] + np.array(cra[i-1, j], cra[i+1, j])/2.
        dec = cdec[i,j] + np.array(cdec[i, j-1], cra[i, j+1])/2.
        dr = ra[corners[0]]
        ddec = dec[corners[1]]
