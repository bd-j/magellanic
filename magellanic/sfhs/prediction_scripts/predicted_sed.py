import os, sys
import numpy as np
import matplotlib.pyplot as pl
pl.rcParams['image.origin'] = 'lower'
pl.rcParams['image.interpolation'] = 'nearest'
pl.rcParams['image.aspect'] = 'equal'
import fsps
from magellanic import  mcutils
from predicted_cmd import load_data, rebin_partial_cmds

background = {'lmc':{'irac_1':1.2e-4, 'irac_2':2e-4, 'irac_3': 2.3e-3, 'irac_4': 1.3e-3},
              'smc':{'irac_1':4e-5, 'irac_2':2.5e-5, 'irac_3':2e-3, 'irac_4':2.0e-4},
              'units': 'maggies/region'
              }
              

def build_predicted_images(cloud, rnames, influxes):
    """
    """
    fluxes = np.atleast_2d(influxes)
    crpix, crval, cdelt, [nx, ny] = mcutils.mc_ast(cloud, badenes=True)
    im = np.zeros([fluxes.shape[0], nx, ny])
    
    for i, rname in enumerate(rnames):
        x, y  = mcutils.regname_to_xy(rname, cloud=cloud)
        im[:, x, y] = fluxes[:, i, None] / np.size(x)
    return im

def read_observed_images(cloud, bands, imdir='.'):
    """
    """
    istring = '{0}/HZ_{1}_{2}.fits'
    import astropy.io.fits as pyfits
    images, headers = [], []
    for b in bands:
        imname = istring.format(imdir, cloud, b)
        images.append(pyfits.getdata(imname))
        headers.append(pyfits.getheader(imname))
        
    return images, headers
       
if __name__ == "__main__":

    # parse user input
    if len(sys.argv) > 1:
        cloud = sys.argv[1].lower()
        bands = sys.argv[2:]
    else:
        cloud = 'smc'
        #Choose bands for which you want to predict images.
        bands = ['irac_1']

        
    # Cloud SFHs
    if cloud == 'smc':
        dm = 18.89
    elif cloud == 'lmc':
        dm=18.49
    mass, rnames, mets, esfh = load_data(cloud)
    
    # Band information
    ab_to_vega = np.atleast_1d(np.squeeze([f.msun_ab-f.msun_vega for b in bands
                             for f in [fsps.get_filter(b)] ]))
    norm = 10**(-0.4 * (dm + ab_to_vega))
        
    #SPS
    sps = fsps.StellarPopulation(compute_vega_mags=True)
    sps.params['sfh'] = 0
    sps.params['imf_type'] = 0
    sps.params['tpagb_norm_type'] = 2 #VCJ
    sps.params['add_agb_dust_model'] = True
    sps.params['agb_dust'] = 1.0
    if len(sps.ssp_ages) == 107:
        isocname = 'MIST_VW'
    else:
        isocname = 'Padova2007'

    # Produce SEDs for each age and Z
    sed = len(mets) * []
    for i,z in enumerate(mets):
        sed.append([])
        # partial SEDs for this metallicity
        sps.params['zmet'] = np.argmin(np.abs(sps.zlegend - z)) + 1
        pseds = sps.get_mags(tage=0, bands=bands)
        ages = sps.ssp_ages
        # convert to linear flux units
        pseds = 10**(-0.4 * pseds)
        # degrade temporal resolution
        rpseds = np.squeeze(rebin_partial_cmds(pseds[:,:, None], ages, esfh)).T
        sed[i] = np.atleast_2d(rpseds).T

    sed = np.array(sed)
    fluxes  = (mass[:,:,:,None] * sed[None, :, : :])
    # Sum over age and metallicity
    tot_fluxes = fluxes.sum(axis=1).sum(axis=1).T
    #sys.exit()
    
    pimages = build_predicted_images(cloud, rnames, tot_fluxes)
    oimages, hdrs = read_observed_images(cloud, bands, imdir='hz_images')

    #### OUTPUT ####

    # plot ratios vs predicted flux
    nplot = int(np.ceil(np.sqrt(len(bands) * 1.0)))
    fig, axes = pl.subplots(nplot, nplot, figsize=(8,6))
    for i, ax in enumerate(axes.flatten()):
        if len(bands) <= i:
            continue
        good = (oimages[i]-background[cloud][bands[i]] > 0) & (pimages[i] > 0)
        oim = oimages[i][good] - background[cloud][bands[i]]
        ratio = (pimages[i][good] * norm[i])/oim
        mratio = np.nanmedian(ratio)
        flux_ratio = (pimages[i][good] * norm[i]).sum()/oim.sum()
        print(bands[i], flux_ratio)
        ax.plot((pimages[i][good] * norm[i]), ((pimages[i][good] * norm[i])/oim), 'o')
        ax.set_ylim(0.1 * mratio, 10 * mratio)
        ax.set_yscale('log')
        ax.axhline(1.0, linestyle='--', color='black')
        ax.axhline(mratio, linestyle=':', color='red', label='Median')
        ax.set_ylabel('Pred/Obs')
        ax.set_xlabel('Predicted Flux (AB maggies)')
        ax.text(0.8, 0.9, '{0}\n median={1:4.3f}'.format(bands[i], mratio), transform=ax.transAxes)
    fig.suptitle('{0} {1} tpgb:{2}'.format(cloud, isocname, sps.params['tpagb_norm_type']))
    #fig.savefig('figures/fluxratio_{0}_{1}_{2}.pdf'.format(cloud, bands[i], isocname))
    fig.show()

    
    
    
