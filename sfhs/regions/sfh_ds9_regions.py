import numpy as np
import sfhutils as utils

cloud = 'lmc'
hznative = True

if hznative:
    hzext = '_hz'
else:
    hzext = ''
    
if cloud.lower() == 'lmc':
    regions = utils.lmc_regions()
    out = open('lmc_regions{}.reg'.format(hzext) ,'w')
    dm = 18.5
    #cr = -68.78333
    dra, ddec =  24./60./np.cos(np.deg2rad(-69.38333)),  24./60.
    minra, mindec = 67.75 - dra/2, -72.3833
elif cloud.lower() == 'smc':
    regions = utils.smc_regions()
    out = open('smc_regions{}.reg'.format(hzext) ,'w')
    dm = 18.9
    
    #the minimum ra in deg of the 'A' corner
    minra, mindec = 6.25, -74.95
    #the dra in deg worked out from figure 3 of H & Z
    dra = 0.5/11 * 15.
    ddec = 12./60.
    
fstring = 'fk5;polygon(' + 7 * '{:10.6f},' + '{:10.6f}) # text={{{}}}\n'
if hznative:
    fstring = fstring.replace('#','# color=red')
    print(fstring)
corners = np.array([1,-1,-1,1]) * dra/2., np.array([-1,-1,1,1]) * ddec/2
allra =[]
alldec = []
rname =[]


for k, v in regions.iteritems():
    #pixel half width in degrees
    if k == 'header':
        continue

    if len(k) > 2:
        sx, sy = float(k[3]), float(k[4])
        factor = 2
    else:
        sx, sy = 0, 0
        factor = 1
        
    loc = v['loc'].split()    
    ddeg = float(loc[3][:-1])
    dec = np.sign(ddeg) * (abs(ddeg) + float(loc[4][:-1])/60.) #use reported coords
    if not hznative:
        dec = mindec + ddec * (ord(k[1]) - ord('A')) + (sy+0.5)*ddec/factor -0.5/2*ddec #build your own coords
    #    dra = 24./60./np.cos(np.deg2rad(dec))

    ra = 15.*( float(loc[1][:-1]) + float(loc[2][:-1])/60.)
    if not hznative:
        ra = minra + dra * (ord(k[0]) - ord('A')) + (sx + 0.5)*dra/factor 

    dr = ra + corners[0]/factor 
    dd = dec + corners[1]/factor
    
    values = np.array(zip(dr, dd)).flatten().tolist() + [k]
    out.write(fstring.format(*values))
    allra += [ra]
    alldec += [dec]
    rname += [k]
    
out.close()
    #corners = ra +

ii = []
for i in range(26):
    try:
        ii += [rname.index('D'+chr(i+65)+'_00')]
        
    except:
        pass
