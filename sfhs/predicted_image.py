#sooooo inefficient
import sys
import numpy as np
import matplotlib.pyplot as pl
import pickle

import astropy.io.fits as pyfits
import regionsed
import fsps

# Methods for reading H&Z data as dictionaries
#  of structured arrays
import sfhutils as utils

cloud, filters = 'smc', ['galex_NUV', 'spitzer_irac_ch1', 'spitzer_irac_ch4', 'spitzer_mips_24']
min_tpagb_age, lf_band, wave, agb_dust = 0.0, '4', '{8\mu m}', 1.0
ldir = 'lf_data/'
outdir = 'tmp/'
#########
# Initialize the import objects (SPS, SFHs, LFs)
#########
# SPS
sps = fsps.StellarPopulation(add_agb_dust_model = True)
sps.params['agb_dust'] = agb_dust
dust = ['nodust', 'agbdust']
sps.params['imf_type'] = 0.0

# SFHs
if cloud.lower() == 'lmc':
    regions = utils.lmc_regions()
    dm = 18.5
    zlist = [7, 11, 13, 16]
    lffiles = ['{0}z{1:02.0f}_tau{2:02.0f}_vega_irac{3}_lf.txt'.format(ldir, z, agb_dust*10, lf_band) for z in zlist]
elif cloud.lower() == 'smc':
    regions = utils.smc_regions()
    dm = 18.9
    zlist = [7, 13, 16]
    lffiles = ['{0}z{1:02.0f}_tau{2:02.0f}_vega_irac{3}_lf.txt'.format(ldir, z, agb_dust*10, lf_band) for z in zlist]
else:
    print('do not understand your MC designation')

# LFs
try:
    lf_bases = [utils.read_lfs(f) for f in lffiles]
    #zero out select ages

    for j, base in enumerate(lf_bases):
        blank = base['ssp_ages'] <= min_tpagb_age
        base['lf'][blank,:] = 0
    
        #plot the lfs to make sure they are ok
        ncolors = base['lf'].shape[0]
        cm = pl.get_cmap('gist_rainbow')
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_color_cycle([cm(1.*i/ncolors) for i in range(ncolors)])
        for i,t in enumerate(base['ssp_ages']):
            ax.plot(base['bins'], base['lf'][i,:], linewidth = 3,
                    label = '{:4.2f}'.format(t), color = cm(1.*i/ncolors))
        ax.legend(loc =0, prop = {'size':6})
        ax.set_ylim(1e-6,3e-4)
        ax.set_yscale('log')
        ax.set_xlabel(r'$M_{}$'.format(wave))
        ax.set_ylabel(r'$n(<M, t)$')
        fig.savefig('{}.png'.format(lffiles[j].replace('.txt','')))
        pl.close(fig)

except(NameError):
    lf_bases = None

    
###############
# Main piece of code to do all the SFH integrations
###############
dat = regionsed.regionsed(regions, sps, lf_bases = lf_bases, filters = filters,)
locs, name, mags, lfs = dat
#bins = lf_basis['bins']
bins = regionsed.lbins
ind_lmax = np.searchsorted(bins, -8.5) -1

#############
# Build output images and LF cubes
############
# Convert pixel letters to numbers and generate output
cell = [( ord(n[0].upper()) - ord('A'), ord(n[1].upper()) - ord('A')) for n in name]
cell = np.array(cell)
# Dilate lmc pixels
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

#############
# Write output
############
            
# Write out images as fits and jpg
for i, f in enumerate(filters):
    fig, ax = pl.subplots(1,1)
    image = ax.imshow(np.log10(im[i,:,:].T), interpolation = 'nearest', origin = 'lower')
    ax.set_title('{0}  @ {1}'.format(cloud.upper(), f))
    ax.set_xlabel('RA (pixels)')
    ax.set_ylabel('Dec (pixels)')
    cbar = pl.colorbar(image, orientation = 'horizontal', shrink = 0.7, pad = 0.12)
    cbar.set_label(r'log F({0}) (AB maggies)'.format(f))
    fstring = '{3}{0}.log_{1}.{2}.harris_zaritsky'.format(cloud, f, dust[int(sps.params['agb_dust'])], outdir)
    pl.savefig(fstring + '.png')
    pl.close(fig)
    pyfits.writeto(fstring + '.fits', im[i,:,:].T, clobber = True)

#write out AGB N(>M) images as fits
for lim in np.arange(-6.7, -9.5, -0.5):
    ind = np.searchsorted(bins, lim) -1
#    pyfits.writeto('test.fits', agb[ind,:,:].T, clobber = True)

#write out AGB N(>M) images as a pickle file
agb_cube = {}
agb_cube['agb_clf_cube'] = agb
agb_cube['mag_bins'] = bins
out = open("{0}clf.{1}.tau{02:2.0f}.irac{3}.p".format(outdir, cloud.lower(), agb_dust*10, lf_band), "wb")
pickle.dump(agb_cube, out)
out.close()

# Plot the total LF
fig, ax = pl.subplots(1,1)
lf_tot = agb.sum(-1).sum(-1)
ax.plot(bins + dm, lf_tot)
ax.set_ylabel(r'$N(<M)$ (total for cloud)')
ax.set_xlabel(r'$m_{}$ (Vega apparent)'.format(wave))
ax.set_title(cloud.upper())
ax.set_yscale('log')
fig.savefig('{0}total_agb_clf.{1}.tau{02:2.0f}.irac{3}.png'.format(outdir, cloud.lower(), agb_dust*10, lf_band))


