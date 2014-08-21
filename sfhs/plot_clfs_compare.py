import matplotlib.pyplot as pl
import numpy as np

clouds =['lmc', 'smc']
dmods = [18.5, 18.9]
isocs = ['MG08', 'CG10','N2', 'N2_test']
tau_agb = [1.0]
bands = ['irac2','irac4']
rdir = 'results_compare/'

icolors = {'MG08': 'blue', 'CG10': 'green', 'N2':'orange', 'N2_test':'purple'}

def readclf(filename):
    f = open(filename, 'r')
    dat = f.readlines()[2:]
    dat = [d.split() for d in dat]
    data = np.array(dat).astype(float)
    return data[:,0], data[:,1]


if __name__ == '__main__':
     
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
