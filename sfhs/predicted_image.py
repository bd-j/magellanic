#sooooo inefficient
import sys
import numpy as np
import matplotlib.pyplot as pl

import astropy.io.fits as pyfits
import regionsed
import fsps

# Methods for reading H&Z data as dictionaries
#  of structured arrays
import sfhutils as utils

cloud, min_tpagb_age, filters = 'lmc', 0, ['galex_NUV', 'spitzer_irac_ch1', 'spitzer_irac_ch4', 'spitzer_mips_24']

#########
# Initialize the import objects (SPS, SFHs, LFs)
#########
# SPS
sps = fsps.StellarPopulation(add_agb_dust_model = True)
sps.params['agb_dust'] = 1
dust = ['nodust', 'agbdust']
sps.params['imf_type'] = 0.0

# SFHs
if cloud.lower() == 'lmc':
    regions = utils.lmc_regions()
    dm = 18.5
    lffile = 'lf_data/irac4_luminosity_function_lmc.txt'
elif cloud.lower() == 'smc':
    regions = utils.smc_regions()
    dm = 18.9
    lffile = 'lf_data/irac4_luminosity_function.txt'

else:
    print('do not understand your MC designation')

# LFs
try:
    lf_basis = utils.read_lfs(lffile)

    #zero out select ages
    blank = lf_basis['ssp_ages'] <= min_tpagb_age
    lf_basis['lf'][blank,:] = 0
    #plot the lfs to make sure they are ok
    ncolors = lf_basis['lf'].shape[0]
    cm = pl.get_cmap('gist_rainbow')
    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1.*i/ncolors) for i in range(ncolors)])
    for i,t in enumerate(lf_basis['ssp_ages']):
        #if t <= 8.6:
        #    continue
        ax.plot(lf_basis['bins'], lf_basis['lf'][i,:], linewidth = 3,
                label = '{:4.2f}'.format(t), color = cm(1.*i/ncolors))
    ax.legend(loc =0, prop = {'size':6})
    ax.set_ylim(1e-6,1e-4)
    ax.set_yscale('log')
    ax.set_xlabel(r'$M_{{8\mu m}}$')
    ax.set_ylabel(r'$n(<M, t)$')
    fig.savefig('agb_8m_lf_{1}.mint{0:.1f}.png'.format(min_tpagb_age, cloud))
    pl.close(fig)

except(NameError):
    lf_basis = None

#sys.exit()

    
#modify the agb LFs by blanking the     
#main piece of code to do all the SFH integrations
dat = regionsed.regionsed(regions, sps, lf_basis = lf_basis, filters = filters,)
locs, name, mags, lfs = dat
bins = lf_basis['bins']
ind_lmax = np.searchsorted(bins, -8.5) -1


#convert pixel letters to numbers and generate output
cell = [( ord(n[0].upper()) - ord('A'), ord(n[1].upper()) - ord('A')) for n in name]
cell = np.array(cell)
#dilate lmc pixels
factor = 2 - int(cloud == 'smc')
nx, ny = (cell[:,0].max()+1), (cell[:,1].max()+1)
im = np.zeros([ len(filters), nx * factor, ny * factor])
agb = np.zeros([ len(bins), nx * factor, ny * factor])

for i, n in enumerate(name):
    x, y = cell[i,:] * factor
    if (len(n) > 2) or (factor == 1):
        if len(n) > 2:
            sx, sy = int(n[3]), int(n[4])
        else:
            sx, sy = 0, 0
        im[:, x+sx, y+sy] = 10**(-0.4 * mags[i,:])
        if lfs is not None:
            agb[:, x+sx, y+sy] = lfs[i,:]
    else:
        im[:, x:x+2, y:y+2] = 10**(-0.4 * mags[i,:, None, None])/4.
        if lfs is not None:
            agb[:, x:x+2, y:y+2] = lfs[i,:, None, None]/4.

#write out images as fits and jpg
for i, f in enumerate(filters):
    fig, ax = pl.subplots(1,1)
    image = ax.imshow(np.log10(im[i,:,:].T), interpolation = 'nearest', origin = 'lower')
    ax.set_title('{0}  @ {1}'.format(cloud.upper(), f))
    ax.set_xlabel('RA (pixels)')
    ax.set_ylabel('Dec (pixels)')
    cbar = pl.colorbar(image, orientation = 'horizontal', shrink = 0.7, pad = 0.12)
    cbar.set_label(r'log F({0}) (AB maggies)'.format(f))
    fstring = '{0}.log_{1}.{2}.harris_zaritsky'.format(cloud, f, dust[int(sps.params['agb_dust'])])
    pl.savefig(fstring + '.png')
    pl.close(fig)
    pyfits.writeto(fstring + '.fits', im[i,:,:].T, clobber = True)

#write out AGB N(>M) images as fits
for lim in np.arange(-6.7, -9.5, -0.5):
    ind = np.searchsorted(bins, lim) -1
    pyfits.writeto('{0}.Nagb_clf.mint{2:0.1f}.M8_AB{1:.1f}.fits'.format(cloud, lim, min_tpagb_age), agb[ind,:,:].T, clobber = True)

#plot the total LF

fig, ax = pl.subplots(1,1)
lf_tot = agb.sum(-1).sum(-1)
ax.plot(bins - 4.4 + dm, lf_tot)
ax.set_ylabel(r'$N(<M)$ (total for cloud)')
ax.set_xlabel(r'$m_{8\mu m}$ (Vega apparent)')
ax.set_title(cloud.upper())
ax.set_yscale('log')
fig.savefig('total_agb_lf_{}.png'.format(cloud.lower()))


#write out AGB N(>M) images as png
ind = np.searchsorted(bins, -6.7) -1

pl.figure()
fig, ax = pl.subplots(1,1)
image = ax.imshow(agb[ind,:,:].T, interpolation = 'nearest', origin = 'lower')
ax.set_title('{0}'.format(cloud.upper()))
ax.set_xlabel('RA (pixels)')
ax.set_ylabel('Dec (pixels)')
cbar = pl.colorbar(image, orientation = 'horizontal', shrink = 0.7, pad = 0.12)
cbar.set_label(r'$N_{{agb}} (M < {0:.2f})$ (per pixel)'.format(bins[ind]))
fstring = '{0}.Nagb_clf.mint{2:0.1f}.M8_AB{1:.1f}'.format(cloud, bins[ind], min_tpagb_age)
pl.savefig(fstring + '.png')
pl.close(fig)
