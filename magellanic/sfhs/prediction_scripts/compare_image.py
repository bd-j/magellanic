import os, sys
import numpy as np
import matplotlib.pyplot as pl
pl.rcParams['image.origin'] = 'lower'
pl.rcParams['image.interpolation'] = 'nearest'
pl.rcParams['image.aspect'] = 'equal'
import fsps
from magellanic import  mcutils
from predicted_cmd import load_data, rebin_partial_cmds


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

def plot_images(pimage, oimage, norm, margin=0.03, bandname='', delta=0):
    fig, axes = pl.subplots(1, 3)
    pl.subplots_adjust(bottom=0.2, hspace=0.3, top=0.95)
    #lpad = fig.subplotpars.left
    #print(type(fig.subplotpars.right), fig.subplotpars.left, margin)
    #width = (fig.subplotpars.right - fig.subplotpars.left)/3 - margin
    caxes = [fig.add_axes([ax.get_position(fig).bounds[0], 0.2,
                           ax.get_position(fig).bounds[2], 0.04])
             for ax in axes]

    im = axes[0].imshow((norm*pimage).T, interpolation='nearest')
    axes[0].set_title('Predicted {}'.format(bandname))
    c0 = im.get_clim()
    fig.colorbar(im, caxes[0], orientation='horizontal')
    
    im = axes[1].imshow((oimage).T, interpolation='nearest')
    axes[1].set_title('Observed {}'.format(bandname))
    im.set_clim(c0 / 10**delta)
    fig.colorbar(im, caxes[1], orientation='horizontal')
    
    im = axes[2].imshow(np.log10(pimage*norm/oimage).T,
                        vmin=-0.5+delta, vmax=1.0+delta,
                        interpolation='nearest')
    axes[2].set_title('log Pred/Obs {}'.format(bandname))
    fig.colorbar(im, caxes[2], orientation='horizontal')
    caxes[-1].set_xlabel('log Pred/Obs {}'.format(bandname))
    
    [ax.set_xlim(ax.get_xlim()[::-1]) for ax in axes]
    return fig, axes, caxes
       
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
    i = 0
    mratio = np.nanmedian((pimages[i] * norm[i])/oimages[i])
    # Plot images
    fig, axes, cax = plot_images(pimages[i], oimages[i], norm[i],
                                 bandname=bands[i], delta=np.log10(mratio))
    fig.suptitle('{0} {1} tpgb_type:{2}'.format(cloud, isocname, sps.params['tpagb_norm_type']))
    fig.show()
    fig.savefig('figures/cimage_{0}_{1}_{2}.pdf'.format(cloud, bands[i], isocname))

    #sys.exit()
    
    # plot ratios vs predicted flux
    fig, ax = pl.subplots()
    ax.plot((pimages[i] * norm[i]).flatten(), ((pimages[i] * norm[i])/oimages[i]).flatten(), 'o')
    ax.set_ylim(0.1 * mratio, 10 * mratio)
    ax.set_yscale('log')
    ax.axhline(1.0, linestyle='--', color='black')
    ax.axhline(mratio, linestyle=':', color='red', label='Median')
    ax.set_ylabel('Pred/Obs')
    ax.set_xlabel('Predicted Flux (AB maggies)')
    ax.text(0.8, 0.9, 'median={:4.3f}'.format(mratio), transform=ax.transAxes)
    fig.suptitle('{0} {1} tpgb:{2}'.format(cloud, isocname, sps.params['tpagb_norm_type']))
    fig.savefig('figures/fluxratio_{0}_{1}_{2}.pdf'.format(cloud, bands[i], isocname))
    fig.show()

    
    
    
