import numpy as np
import matplotlib.pyplot as pl
import hmc
import mcutils

#theta = np.zeros(nage)

class LinearModel(object):

    def __init__(self, mass, N):
        self.mass = mass
        self.N = N
        self.lower = 0.0
        self.upper = np.inf
    
    def lnprob(self, theta):
        Nex = np.dot(theta, self.mass)
        lnp = self.N * np.log(Nex) - Nex
        return lnp[np.isfinite(lnp)].sum()

    def lnprob_grad(self, theta):
        Nex = np.dot(theta, self.mass)
        nonzero = Nex > 0
        grad_lnp = np.dot((self.N/Nex - 1)[nonzero], self.mass.T[nonzero,:])
        return grad_lnp
    
    def check_constrained(self, theta):
        """Method that checks the value of theta against constraints.
        If theta is above or below the boundaries, the sign of the momentum
        is flipped and theta is adjusted as if the trajectory had
        bounced off the constraint. Returns the new theta vector, a
        vector of multiplicative signs for the momenta, and a flag for
        if the values are still out of bounds."""

        #initially no flips
        sign = np.ones_like(theta)
        oob = True #pretend we started out-of-bounds to force at least one check
        #print('theta_in ={0}'.format(theta))
        while oob:
            above = theta > self.upper
            theta[above] = 2*self.upper - theta[above]
            sign[above] *= -1
            below = theta < self.lower
            theta[below] = 2*self.lower - theta[below]
            sign[below] *= -1
            oob = np.any(below | above)
            #print('theta_out ={0}'.format(theta))
        return theta, sign, oob

def load_data(cloud):
    """
    :returns name:
        list of length (nreg,)

    :returns mass:
        ndarray of masses formed, of shape (nage, nreg)

    :returns N:
        ndarray giving number of stars observed in each region, of
        shape (nreg)
    """
    import datautils
    import sedpy.ds9region as ds9

    #SFH data
    if cloud.lower() is 'lmc':
        regions = mcutils.lmc_regions()
    else:
        regions = mcutils.smc_regions()
    regions.pop('header')

    #AGB data
    cat, cols = datautils.cloud_cat(cloud)
    
    #mass = np.zeros([nage, nreg])
    #N = np.zeros(nreg)
    mass, N = [], []
    for name, dat in regions.iteritems():
        shape, defstring = mcutils.corners_of_region(name, cloud, string=True)
        ds9reg = ds9.Polygon(defstring)
        acat = datautils.select(cat, cols, ds9reg, codes=cols['agb_codes'])
        N.append( len(acat) )
        total_sfh = None
        for s in dat['sfhs']:
            total_sfh = mcutils.sum_sfhs(total_sfh, s)
        mass += [ total_sfh['sfr'] * (10**total_sfh['t2'] - 10**total_sfh['t1']) ]

    example_sfh = total_sfh
    
    return regions.keys(), np.array(mass).T, np.array(N), example_sfh

if __name__ == "__main__":

    cloud = 'lmc'
    rname, mass, N, esfh = load_data(cloud)
    nage, nreg = mass.shape
    time = (esfh['t1'] + esfh['t2'])/2
    
    print('loaded data')
    
    model = LinearModel(mass, N)
    hsampler = hmc.BasicHMC()
    # Initial guess is that all age bins contribute equally
    initial = N.sum()/mass.sum()/nage
    initial = np.zeros(nage) +initial

    # Do some L-BFGS-B minimization?
    
    # HMC sampling

    nsegmax = 10
    iterations = 50
    length = 100
    alleps=[]
    
    print('starting sampling')
    pos, prob, thiseps = hsampler.sample(initial, model, iterations=10, epsilon=None,
                                    length=length, store_trajectories=False, nadapt=0)
    eps = thiseps

    for k in range(nsegmax):
        #advance each sampler after adjusting step size
        afrac = hsampler.accepted.sum()*1.0/hsampler.chain.shape[0]
        if afrac >= 0.9:
            shrink = 2.0
        elif afrac <= 0.6:
            shrink = 1/2.0
        else:
            shrink = 1.00
        eps *= shrink
        pos, prob, thiseps = hsampler.sample(hsampler.chain[-1,:], model,
                                             iterations=iterations,
                                             epsilon=eps, length=length,
                                             store_trajectories=False, nadapt=0)
    alleps.append(thiseps) #this should not have actually changed during the sampling

    hsampler.sample(hsampler.chain[-1,:], model, iterations=10000, length=100,
                    epsilon=eps, store_trajectories=True)


    ptiles = np.percentile(hsampler.chain, [16, 50, 84], axis=0)
    median, minus, plus = ptiles[1,:], ptiles[1,:] - ptiles[0,:], ptiles[2,:] - ptiles[1,:]
    efig, eaxes = pl.subplots()
    eaxes.errorbar(time, median,
                   yerr=np.vstack([minus, plus]),
                   color='blue', marker='o', mew=0.)
    eaxes.set_xlabel(r'$\log \, t_j ($yrs$)$')
    eaxes.set_ylabel(r'$\theta_j \, (AGB \#/M_\odot) \, ($specific frequency$)$')
    eaxes.set_title(cloud.upper())

    clr = 'darkcyan'
    bfig, baxes = pl.subplots()
    bp = baxes.boxplot(hsampler.chain,  labels = [str(t) for t in time],
                       whis=[16, 84], widths=0.9,
                       boxprops = {'alpha': 0.3, 'color':clr},
                       whiskerprops = {'linestyle':'-', 'linewidth':2, 'color':'black'},
                       showcaps=False, showfliers=False, patch_artist=True)
    baxes.set_xlabel(r'$\log \, t_j ($yrs$)$', labelpad=15)
    baxes.set_ylabel(r'$\theta_j \, (AGB \#/M_\odot) \, ($specific frequency$)$')
    baxes.set_title(cloud.upper())

    bfig.show()
    efig.show()
    bfig.savefig('{0}_theta.pdf'.format(cloud.lower()))
