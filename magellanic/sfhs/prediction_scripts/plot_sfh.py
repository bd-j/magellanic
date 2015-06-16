import numpy as np
import matplotlib.pyplot as pl
from predicted_cmd import *


cloud, dm = 'lmc', 18.49


fig, axes = pl.subplots(2, 1)
fig.subplots_adjust(bottom=0.2, hspace=0.3, top=0.95)
cax = fig.add_axes([0.2, 0.06, 0.6, 0.04])

images = []
for ax, cloud in zip(axes, ['lmc', 'smc']):
    mass, _, mets, sfh = load_data(cloud)
    dt = 10**sfh['t2'] - 10**sfh['t1']
    tlabel = ['{:4.2f}'.format((s['t1']+s['t2'])/2) for s in sfh]
    sfr = mass / dt[None, None, :]
    
#    images.append(ax.imshow(np.log10(sfr.sum(axis=0)), interpolation='nearest',
#                aspect='auto', cmap=pl.cm.gray_r))
    images.append(ax.imshow(np.log10(mass.sum(axis=0)), interpolation='nearest',
                aspect='auto', cmap=pl.cm.gray_r))
    ax.set_xticks(np.arange(len(sfh)))
    ax.set_xticklabels(tlabel, fontsize=8)
    ax.set_yticks(np.arange(len(mets)))
    ax.set_yticklabels(['{:5.4f}'.format(z) for z in mets])
    ax.set_ylabel('Z')
    ax.set_xlabel('log Age', fontsize=10)
    ax.text(0.1, 0.9, cloud.upper(), transform=ax.transAxes, fontsize=14, color='red')
    ax.xaxis.grid()
    ax.yaxis.grid()
fig.colorbar(images[0], cax, orientation='horizontal')
[im.set_clim(images[0].get_clim()) for im in images]
cax.set_xlabel('log M$_{{formed}}$(Z, Age)', fontsize=10)
fig.show()
