import numpy as np
import matplotlib
#matplotlib.use('Tkagg')
import matplotlib.pyplot as pl
import sys, pickle, hmc
from magellanic import mcutils

#theta = np.zeros(nage)

class LinearModel(object):

    def __init__(self, mass, N):
        self.mass = mass
        self.N = N
        self.lower = 0.0
        self.upper = np.inf

    def expected_number(self, theta):
        return np.dot(theta, self.mass)
    
    def lnprob(self, theta):
        Nex = self.expected_number(theta)
        lnp = self.N * np.log(Nex) - Nex
        return lnp[np.isfinite(lnp)].sum()

    def lnprob_grad(self, theta):
        Nex = self.expected_number(theta)
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

def load_data(cloud, agbtype=None, **kwargs):
    """
    :returns name:
        list of length (nreg,)

    :returns mass:
        ndarray of masses formed, of shape (nage, nreg)

    :returns N:
        ndarray giving number of stars observed in each region, of
        shape (nreg)
    """
    from magellanic import datautils
    import sedpy.ds9region as ds9

    #SFH data
    if cloud.lower() == 'lmc':
        print('doing lmc')
        regions = mcutils.lmc_regions()
    else:
        print('doing smc')
        regions = mcutils.smc_regions()
    regions.pop('header')

    #AGB data
    cat, cols = datautils.cloud_cat(cloud.lower())
    agbcat = datautils.select(cat, cols, codes=cols['agb_codes'], **kwargs)
    agbregion = ds9.Polygon(datautils.bounding_hull(agbcat, cols)[0])
    if agbtype is None:
        codes = cols['agb_codes']
    else:
        codes = [code for code in cols['agb_codes']
                 if agbtype(code)]

    # loop over regions and count agb stars
    mass, N, rnames = [], [], []
    for name, dat in regions.iteritems():
        shape, defstring = mcutils.corners_of_region(name, cloud.lower(), string=True, **kwargs)
        reg = ds9.Polygon(defstring)
        if not np.all(agbregion.contains(reg.ra, reg.dec)):
            continue
        acat = datautils.select(agbcat, cols, region=reg, codes=codes, **kwargs)
        N.append( len(acat) )
        total_sfh = None
        for s in dat['sfhs']:
            total_sfh = mcutils.sum_sfhs(total_sfh, s)
        mass += [ total_sfh['sfr'] * (10**total_sfh['t2'] - 10**total_sfh['t1']) ]
        rnames.append(name)
        
    example_sfh = total_sfh    
    return rnames, np.array(mass).T, np.array(N), example_sfh


def run_hmc(initial, model, length=100, nsegmax=20,
            iterations=5000, adapt_iterations=100,
            verbose=False):
    """Run an HMC sampler for the given number of steps after a
    certain amount of adaptation.  returns the sampler object.
    """
    hsampler = hmc.BasicHMC(verbose=verbose)
    pos, prob, teps = hsampler.sample(initial, model, iterations=10, epsilon=None,
                                     length=length, store_trajectories=False, nadapt=0)
    eps = teps
    
    # Adapt epsilon while burning-in
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
        pos, prob, teps = hsampler.sample(hsampler.chain[-1,:], model, epsilon=eps, length=length,
                                          iterations=adapt_iterations, store_trajectories=False, nadapt=0)

    # Production
    hsampler.sample(hsampler.chain[-1,:], model, iterations=iterations, length=length,
                    epsilon=eps, store_trajectories=True)

    return hsampler


if __name__ == "__main__":

    cloud = sys.argv[1].lower()  #'lmc' | 'smc'
    try:
        bias = bool(sys.argv[2])
        if sys.argv[2] in ['False', 'false']:
            bias = False
    except:
        bias = False
        print('not adding bias')
    #extralabel, agbsel = '_CX', lambda x: ('C' in x) or ('X' in x.upper())
    #extralabel, agbsel = '_C', lambda x: ('C' in x) 
    #extralabel, agbsel = '_X', lambda x: ('X' in x.upper()) 
    #extralabel, agbsel = '_O', lambda x: ('O' in x.upper())
    #extralabel, agbsel = '_All_cb_noRS', None
    extralabel, agbsel = '_All', None
    
    rname, mass, N, esfh = load_data(cloud, agbtype=agbsel, badenes=True)
    time = (esfh['t1'] + esfh['t2']) / 2
    nage, nreg = mass.shape

    if bias:
        #Add a constant offset term
        baseline = np.zeros(nreg) + mass.sum()/nreg/nage
        mass = np.concatenate([mass, baseline[None, :]])
        time = np.array(time.tolist()+[time[-1]])
    
    nage, nreg = mass.shape
    
    # Set up the model and initial position
    model = LinearModel(mass, N)
    
    # Initial guess is that all age bins contribute equally
    initial = N.sum()/mass.sum()/nage
    initial = np.zeros(nage) +initial
    initial = initial * np.random.uniform(1,0.001, size=len(initial))
    if bias:
        initial[-1] = N.sum()/nreg/2.

    # Initial guess from badenes posterior
    badenes_file = 'tex/badenes_results/LMC_MCMC_DTD_AGB_Unbinned_Iter000.dat'
    bdat = np.loadtxt(badenes_file, skiprows=1)
    #initial = bdat[:,3]

    # Do some L-BFGS-B minimization?

    # HMC
    hsampler = run_hmc(initial, model, length=100, nsegmax=20,
                       iterations=5000, adapt_iterations=100)

    #Assemble and write output
    ptiles = np.percentile(hsampler.chain, [16, 50, 84], axis=0)
    median, minus, plus = ptiles[1,:], ptiles[1,:] - ptiles[0,:], ptiles[2,:] - ptiles[1,:]
    maxapost = hsampler.lnprob.argmax()

    output = {'chain':hsampler.chain,
             'lnprob':hsampler.lnprob,
             'time':time,
             'esfh':esfh,
             'cloud':cloud,
             'mass':mass,
             'N': N,
             'bias':bias}

    #tfig = triangle.corner(hsampler.chain)
    with open('chains/{0}{1}_chain.p'.format(cloud.lower(), extralabel), 'wb') as f:
        pickle.dump(output, f)

    fmt = '{:8s} {:7.0f}   ' + nage * ' {:<8.1f}'+'\n'
    with open('chains/{0}{1}_data.dat'.format(cloud.lower(),extralabel),'w') as f:
        f.write('Region     N_agb   '+'  '.join(['m_{}'.format(i) for i in range(nage)])+ '\n')
        for i,rn in enumerate(rname):
            f.write(fmt.format(rn, N[i], *mass[:,i]))

