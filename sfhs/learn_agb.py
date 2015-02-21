import hmc

mass = np.zeros([nage, nreg])
N = np.zeros(nreg)
theta = np.zeros(nage)

class LinearModel(object):

    def __init__(self, mass, N):
        self.mass = mass
        self.N = N
    
    def lnprob(self, theta):
        Nex = np.dot(theta, self.mass)
        lnp = self.N * np.log(Nex) - Nex
        return lnp.sum()

    def lnprob_grad(self, theta):
        Nex = np.dot(theta, self.mass)
        grad_lnp = np.dot((self.N/Nex - 1), self.mass.T)
        return grad_lnp


if __name__ == "__main__":

    model = LinearModel(mass, N)
    hsampler = hmc.BasicHMC()
    # Initial guess is that all ages contribute equally
    initial = mass.sum()/(nreg*nage)
    initial = np.zeros(nage) +initial

    # Do some L-BFGS-B minimization?
    
    # HMC sampling
    hsampler.sample(inital, model, iterations=100, length=10,
                    store_trajectories=True)
