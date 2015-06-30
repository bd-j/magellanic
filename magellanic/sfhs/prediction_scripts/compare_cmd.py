import sys
import numpy as np
import matplotlib.pyplot as pl

import fsps
from magellanic import cmdutils, sputils, mcutils
from predicted_cmd import *

# Define a function for selecting certain "stars" from the
# isochrone.  Only stars selected by this function will contribute
# to the CMD
def select_function(isochrones, cloud=None, **extras):
    """Select only certain stars from the isochrones.
    """
    c, o, b, x = cmdutils.boyer_cmd_classes(isochrones, cloud=cloud,
                                            **extras)
    ## No filtering
    #select, stitle = np.ones(len(isochrones), dtype= bool), 'All stars'
    ## Cstars only
    #select, stitle = (b | x) & (~o) & (isochrones['phase'] == 5), 'CMD cuts for CX'
    ## Phase selction of C stars
    select, stitle = ((isochrones['phase'] == 5) &
                      (isochrones['composition'] >= 1.0)), 'Phase=5, C/O$\geq$1'
    return isochrones[select], stitle

isocname = 'MIST_VW'
cutname = 'C_phasecut'

if __name__ == "__main__":

    #SPS
    sps = fsps.StellarPopulation(compute_vega_mags=True)
    sps.params['sfh'] = 0
    sps.params['imf_type'] = 0
    sps.params['tpagb_norm_type'] = 2 #VCJ
    sps.params['add_agb_dust_model'] = True
    sps.params['agb_dust'] = 1.0

    # Cloud SFHs
    #cloud, dm = 'smc', 18.89
    cloud, dm = 'lmc', 18.49
    mass, rnames, mets, esfh = load_data(cloud)
    #adjust time bins to check for edge effects
    tweak = 0
    esfh['t1'] = np.log10(10**esfh['t1'] + tweak)
    esfh['t2'] = np.log10(10**esfh['t2'] + tweak)
    
    # Choose colors, mags as lists of lists.  The inner lists should
    # be passable as ``color`` and ``mag`` to the make_cmd() method
    # defined above.
    # Here we are doing j and ks but with the same j-k color in both
    # cases.
    colors = [['2mass_j', '2mass_ks', np.linspace(-1, 4.0, 501) ]]
    mags = [['2mass_ks', np.linspace(7, 14, 281) - dm]]

    # Build partial CMDs for each metallicity and project them onto
    # the SFH for that metallicity.  Outer loop should always be
    # metallicity to avoid excessive calls to sps.isochrones(), which
    # are expensive.
    cmd = []
    for i,z in enumerate(mets):
        cmd.append([])
        # Total (summed over regions) cloud SFH for this metallicity
        mtot = mass[:,i,:].sum(axis=0)
        
        # Isochrone data for this metallicity
        sps.params['zmet'] = np.argmin(np.abs(sps.zlegend - z)) + 1
        full_isoc = sps.isochrones()

        # Here you can loop over anything that doesn't affect the
        # generation of the isochrone data to make cmds
        for color, mag in zip(colors, mags):
            #filter the isochrone
            isoc_dat, stitle = select_function(full_isoc, cloud=cloud)
            pcmds, ages = cmdutils.partial_cmds(isoc_dat, tuple(color), tuple(mag))
            if len(ages) == 0:
                cmd
            rpcmds = rebin_partial_cmds(pcmds, ages, esfh)
            #cmd_zc = make_cmd(isoc_dat, tuple(color), tuple(mag), mtot, esfh)
            cmd[i].append(rpcmds)

    # This makes an array of shape (nmet, nband, ncolor-1, nmag-1)
    # NB: this will not work if there are not the same number of bins
    # in each color or in each mag, though you can still use ``cmd``
    # in list form in that case.
    cmd = np.squeeze(np.array(cmd))

    #sys.exit()
    
    #plot AGB numbers
    sfreq = cmd.sum(axis=-1).sum(axis=-1)
    fig, ax = pl.subplots()
    ax.plot(esfh['t1'], sfreq.sum(axis=0), '-o', label='total')
    for i, z in enumerate(mets):
        ax.plot(esfh['t1'], sfreq[i,:], '-o', label='Z={:4.3}'.format(z))
    ax.set_ylabel('N$_{agb}/\mathrm{M}_\odot$')    
    ax.set_title('MIST, '+ stitle)
    ax.legend(loc=0)
    ax.set_xlabel('$\log Age$')
    fig.savefig('figures/sagb_{0}_{1}.pdf'.format(isocname, cutname))

    fig.show()

    # AGB numbers weighted by SFH
    fig, ax = pl.subplots(figsize=(10,5))
    fig.subplots_adjust(bottom=0.2, hspace=0.3, top=0.95)
    cax = fig.add_axes([0.2, 0.06, 0.6, 0.04])
    im = ax.imshow(np.log10(mass.sum(axis=0) * sfreq),
                   interpolation='nearest',aspect='auto', cmap=pl.cm.gray_r)
    tlabel = ['{:4.2f}'.format((s['t1']+s['t2'])/2) for s in esfh]
    ax.set_xticks(np.arange(len(esfh)))
    ax.set_xticklabels(tlabel, fontsize=8)
    ax.set_yticks(np.arange(len(mets)))
    ax.set_yticklabels(['{:5.4f}'.format(z) for z in mets])
    ax.set_ylabel('Z')
    ax.set_xlabel('log Age', fontsize=10)
    tstring = '{0}, {1}, {2}'.format(cloud.upper(), isocname, stitle)
    ax.text(0.05, 0.9, tstring , transform=ax.transAxes, fontsize=14, color='red')
    ax.xaxis.grid()
    ax.yaxis.grid()
    fig.colorbar(im, cax, orientation='horizontal')
    cax.set_xlabel('$\log N_{{agb}}(Age, Z)$')
    fig.savefig('figures/agbfreq_{0}_{1}_{2}.pdf'.format(cloud, isocname, cutname))
    
    jk = np.arange(-1, 4, 0.1)
    k0, k1, k2 = cmdutils.cioni_klines(jk, cloud=cloud)
    # plot output partial cmds
    from matplotlib.backends.backend_pdf import PdfPages
    partialfig = PdfPages('figures/partialcmds_{0}_{1}_{2}.pdf'.format(cloud, isocname, cutname))
    for i, asfh in enumerate(esfh):
        bfig, bax = pl.subplots(2,2)
        for j, (z, ax) in enumerate(zip(mets, bax.flatten())):
            ax.imshow(np.log10(cmd[j,i,:,:].T), interpolation='nearest',
                      extent=[color[-1].min(), color[-1].max(),
                              mag[-1].max()+dm, mag[-1].min()+dm],
                              aspect='auto')
            ax.set_title('Z={}'.format(z), fontsize=8)
            ax.set_xlabel('{0}-{1}'.format(colors[0][0], colors[0][1]))
            ax.set_ylabel('{0}'.format(mags[0][0]))
            ax.set_ylim(13, 9)
            ax.set_xlim(0, 3)
            ax.plot(jk, k0, 'k')
            ax.plot(jk, k1, 'k')
            ax.plot(jk, k2, 'k')
            
        bfig.suptitle('logt={0}-{1}'.format(asfh['t1'], asfh['t2']))
        bfig.subplots_adjust(left=0.05, right=0.95, wspace=0.08, hspace=0.08)
        if i > 4:
            bfig.savefig(partialfig, format='pdf')
        pl.close(bfig)
    partialfig.close()
    sys.exit()
    
    ##### OUTPUT #######
    
    for j in range(4):
        for k in range(16):
            plot(clr[200:300], cmd[j,k,200:300,:].sum(axis=-1),
                 label='{0},{1}'.format(j,k))
            
    # Plot output CMDs
    cfig, caxes = pl.subplots( 1, len(mags) )
    for j, (color, mag) in enumerate(zip(colors, mags)):
        ax = caxes.flat[j]
        im = ax.imshow(np.log10(cmd[:,j,...].sum(axis=0).T), interpolation='nearest',
                       extent=[color[-1].min(), color[-1].max(),
                               mag[-1].max()+dm, mag[-1].min()+dm])
        ax.set_xlabel('{0} - {1}'.format(color[0], color[1]))
        ax.set_ylabel('{0}'.format(mag[0]))
        ax.set_title(cloud.upper())
    cfig.show()


    ##### Observed CMD #####
    from magellanic.sfhs.datautils import cloud_cat, catalog_to_cmd
    from copy import deepcopy
    cat, cols = cloud_cat(cloud)
    ofig, oaxes = pl.subplots( 1, len(mags) )
    for j, (color, mag) in enumerate(zip(colors, mags)):
        ax = oaxes.flat[j]
        appmag = deepcopy(mag)
        appmag[-1] += dm
        ocmd = catalog_to_cmd(cat, color, appmag, catcols=cols)
        im = ax.imshow(np.log10(ocmd.T), interpolation='nearest',
                       extent=[color[-1].min(), color[-1].max(),
                               appmag[-1].max(), appmag[-1].min()])
        ax.set_xlabel('{0} - {1}'.format(color[0], color[1]))
        ax.set_ylabel('{0}'.format(mag[0]))
        ax.set_title('Obs ' + cloud.upper())
    ofig.show()

    
    
