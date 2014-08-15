import numpy as np

def read_galex_tilelist(filename):
    f = open(filename,'r')
    lines = f.readlines()
    f.close()
    nline = len(lines)
    bnum = 0
    
    ttype = ['header','cal_new', 'spec_gr7_new','spec_gr6_pluset','tile_gr7_new',
             'tile_gr7_new_css', 'tile_gr6_pluset','tile_gr6_css','tile_gr6']

    dt = {'names':('survey','tilename','tilenum',
                   'subgrid','ra_cent','dec_cent',
                   'nuv_exptime','fuv_exptime','release'),
          'formats':('a3','a32','a5','a4','<f8','<f8','<f8','<f8','a20')}
    galex = np.zeros(nline,dtype = dt)
    breaks = (np.where(np.char.find(lines, 'Listing') >= 0)) [0]
#print(lines[breaks[0]].split())
#   print(lines[breaks[0]].split()[0] == 'Listing')
#    raise ValueError('stop')
    for i,l in enumerate(lines):
        s = l.split()
        #print(type(s), s, len(s))
        if len(s) == 0 : continue
        if s[0] in ['CAI','AIS','MIS','DIS','GII','NGS', 'ETS','GIS']:
            galex[i] = (s[0], s[1], s[2], s[3], float(s[4]), float(s[5]),
                    float(s[6]),float(s[7]), ttype[bnum])
        elif s[0] == 'Listing':
            bnum += 1
            #print(i,s[:7],ttype[bnum])

    return galex

