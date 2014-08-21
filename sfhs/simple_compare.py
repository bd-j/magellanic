import matplotlib.pyplot as pl
import numpy as np
import pickle
import astropy.io.fits as pyfits
from sedpy import ds9region as ds9reg

smccols = {'RA': 'RAJ2000',
           'DEC': 'DEJ2000',
           'IRAC4': '__8_0_',
           'IRAC2': '__4_5_',
           'STARTYPE': 'Class',
           'agb_codes': ['C-AGB', 'O-AGB', 'x-AGB', 'aO-AGB','FIR']
          }
lmccols = {'RA': 'RA',
           'DEC': 'DEC',
           'IRAC4': 'IRAC4',
           'IRAC2': 'IRAC2',
           'STARTYPE': 'FLAG',
           'agb_codes': ['C', 'O', 'X', 'aO/C', 'aO/O', 'RS-C', 'RS-O', 'RS-X', 'RS-a']
           }

rdir = '/Users/bjohnson/Projects/magellanic/sfhs/results_predicted/'


def select(catalog, coldict, region, codes=None):
    x, y = catalog[coldict['RA']], catalog[coldict['DEC']]
    sel = region.contains(x, y)
    if codes is not None:
        typesel = False
        for c in codes:
            typesel |= (catalog[coldict['STARTYPE']] == c)
        sel = sel & typesel
    return catalog[sel]

    
def cumulative_lf(catalog, bandname):
    mag = catalog[bandname]
    mag = mag[np.isfinite(mag)]
    order = np.argsort(mag)
    return mag[order], np.arange(len(mag))


def cloud_corners(cloud):
    """
    Return strings defining vertices of the polygon enclosing the MCPS
    survey.  These are very approximate.
    """
    c =cloud.lower()
    if c == 'smc':
        corners = '6.25,-74.95,19.0,-74.95,19.0,-70.533,6.25,-70.533'
    elif c == 'lmc':
        corners = '70.0,-72.2,92,-72.2,90,-65.4,72.5,-65.4'
    return corners

    
def cloud_cat(cloud):
       
    c = cloud.lower()
    catname = '/Users/bjohnson/Projects/magellanic/catalog/boyer11_{}.fits.gz'.format(c)
    catalog = pyfits.getdata(catname)
    if c == 'smc':
        cols = smccols
    elif c == 'lmc':
        cols = lmccols
    return catalog, cols


def plot_clf(obs_clf, pred_clf, cloud, band, agb_dust=1.0):
    if cloud.lower() == 'smc':
        dm = 18.9
    else:
        dm = 18.5
        
    fig, ax = pl.subplots(1,1)
    ax.plot(obs_clf[0], obs_clf[1], color = 'red',
            label = 'SAGE observed AGBs')
    ax.plot(pred_clf[0] + dm, pred_clf[1], color = 'blue',
            label = r'Predicted AGBs, $\tau_{{agb}}={0:3.1f}$'.format(agb_dust))
    ax.set_yscale('log')
    ax.set_title('{0} AGBs @ {1}'.format(cloud.upper(), band))
    ax.set_xlabel('magnitude ({0}, apparent, Vega)'.format(band))
    ax.set_ylabel(r'$N(<m)$')
    ax.set_xlim(4,14)
    ax.set_ylim(1, 1e5)
    ax.legend(loc =0)
    return fig, ax

def write_clf(wclf, filename, lftype):
    out = open(filename,'w')
    out.write('{0}\n mag  N<m\n'.format(lftype))
    for m,n in zip(wclf[0], wclf[1]):
        out.write('{0}   {1}\n'.format(m,n))
    out.close()
    

if __name__ == '__main__':

    cloud, agb_dust = 'smc', 1.0
    bands = ['IRAC2', 'IRAC4']
    # Get the observed CLFs
    defstring = cloud_corners(cloud)
    region = ds9reg.Polygon(defstring)
    cat, cols = cloud_cat(cloud)
    subcat = select(cat, cols, region, codes=cols['agb_codes'])
    obs_clfs = [cumulative_lf(subcat, cols[band]) for band in bands]

    # Get the modeled CLFs
    pred_clfs = []
    #the pickle files with the agb CLF cubes
    pfiles = [rdir + 'clf.{0}.tau{2:02.0f}.{1}.p'.format(cloud, b.lower(), agb_dust*10) for b in bands]
    for i,fname in enumerate(pfiles):
        f = open(fname, 'rb')
        agb_cube = pickle.load(f)
        f.close()
        pred_clfs += [(agb_cube['mag_bins'], agb_cube['agb_clf_cube'].sum(-1).sum(-1))]
        
        fig, ax = plot_clf(obs_clfs[i], pred_clfs[i], cloud, bands[i], agb_dust=agb_dust)
        fstring = 'results_compare/clf.{0}.{1}.tau{2:02.0f}'
        values = [cloud, bands[i].lower(), agb_dust*10]
        fig.savefig(fstring.format(*values)+'.png')
        pl.close(fig)
        
        write_clf(pred_clfs[i], fstring.format(*values)+'.dat', 'Predicted')
        fstring = 'results_compare/obs_clf.{0}.{1}'
        write_clf(obs_clfs[i], fstring.format(*values[0:2])+'.dat', 'Observed')
