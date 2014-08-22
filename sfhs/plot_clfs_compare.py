import matplotlib.pyplot as pl
import numpy as np

clouds =['lmc', 'smc']
dmods = [18.5, 18.9]
isocs = ['MG08', 'CG10','N2', 'Basti_N2']
tau_agb = [1.0]
bands = ['irac2','irac4']
rdir = 'results_compare/'

icolors = {'MG08': 'blue', 'CG10': 'green', 'N2':'orange', 'N2_test':'purple', 'Basti_N2': 'magenta'}

def readclf(filename):
    f = open(filename, 'r')
    dat = f.readlines()[2:]
    dat = [d.split() for d in dat]
    data = np.array(dat).astype(float)
    return data[:,0], data[:,1]

def plot_lf(base, thin=1):
    """
    Plot the interpolated input lfs to make sure they are ok
    """
    ncolors = base['lf'].shape[0]
    cm = pl.get_cmap('gist_rainbow')
    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1.*i/ncolors) for i in range(ncolors)])
    for i,t in enumerate(base['ssp_ages']):
        if (i % thin) == 0:
            ax.plot(base['bins'], base['lf'][i,:], linewidth = 3,
                    label = '{:4.2f}'.format(t), color = cm(1.*i/ncolors))
    ax.legend(loc =0, prop = {'size':6})
    ax.set_ylim(1e-6,3e-4)
    ax.set_yscale('log')
    ax.set_xlabel(r'$M_{}$'.format(wave))
    ax.set_ylabel(r'$n(<M, t)$')
    fig.savefig('{}.png'.format(lffile.replace('.txt','')))
    return fig, ax
    
def compare_total():
    
    for cloud, dm in zip(clouds, dmods):
        for band in bands:
            fig, ax = pl.subplots(1,1)
            ax.set_title('{0} AGBs @ {1}'.format(cloud.upper(), band))
            ax.set_xlabel('magnitude ({0}, Vega, apparent)'.format(band))
            ax.set_ylabel(r'$N (<m)$')
            ax.set_xlim(4,14)
            ax.set_ylim(1,1e5)
            ax.set_yscale('log')

            obsf = rdir + '{0}/obs_clf.{1}.{2}.dat'.format('CG10', cloud, band)
            obins, oclf = readclf(obsf)
            ax.plot(obins, oclf, color='red',
                    label=r'observed', linewidth=3.0)
        
            for i,tau in enumerate(tau_agb):
                linestyle = '-'
                for isoc in isocs:
                    if (band=='irac4') and isoc=='CG10':
                        if tau==0.5: tau_file=1.0
                        if tau==1.0: tau_file=0.5
                    else:
                        tau_file = tau
                    
                    label = r'$\tau={0}$, {1}'.format(tau, isoc)
                    if tau == 0.5:
                        linestyle = '--'
    
                    predf = rdir + '{0}/clf.{1}.{2}.tau{3:02.0f}.dat'.format(isoc, cloud, band, tau_file*10)
                    pbin, pclf = readclf(predf)
                    ax.plot(pbin+dm, pclf, label=label,
                            color=icolors[isoc], linestyle=linestyle,
                            linewidth=3)
                    
            ax.legend(loc=0)
            fig.savefig(rdir + 'compare_clfs.{0}.{1}.png'.format(cloud,band))
            pl.close(fig)

def compare_to_ssps(lffile, predfile):
    lf_base = read_lfs(lffile)
    fig, ax = plot_lf(lf_base, wlengths[lf_band], lffile)
    obins, oclf = readclf(obsfile)
    pbins, pclf = readclf(predfile)
    
if __name__ == '__main__':
    compare_total()
