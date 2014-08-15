import os, time
import numpy as np
import matplotlib.pyplot as pl

import starfitter as sf
import extinction, observate
import plotter
import catio

datadir = '/Users/bjohnson/DATA/magellanic/catalogs/LMC/'

#### SETTINGS
#  Distance in Mpc, number of SEDs in grid, min and max wavelength for logL calculation
rp = {'dist': 0.050, 'ngrid': 20e3, 'wave_min':92, 'wave_max':1e7}
# Percentiles of the CDF to return
rp['percentiles'] = np.array([0.025, 0.5, 0.975])
# Input optical catalog(s)
rp['source'] = 'mcps'  #massey|mcps

# Optical catalog parameters
if rp['source'] is 'mcps':
    #simple read of MCPS
    rp['catalog_name'] = datadir + 'mcps/mcps_lmc_V20_detect4.dat'
    rp['fsize'] = 5.8e6
    # Number of entries from MCPS catalog to fit
    rp['nlines'] = 50000
    # Specific lines from the MCPS catalog to fit (it's big!). if None, randomly sample the catalog (Slow!!!)
    rp['lines'] = None # np.arange(1e6)+rp['nlines']
    #SQL selection from the MCPS db
    rp['catalog_name'] = datadir + 'mcps/mcps_lmc.db'
    rp['RAh range'] = (5.5, 5.55)
    rp['Dec range'] = (-67.5, -67.3)
    rp['select_clause'] = ("WHERE bessell_V < 20 AND bessell_V > 0 " +
                           "AND RAh > {0} AND RAh < {1} " +
                           "AND Dec > {2} AND Dec < {3} " +
                           "AND bessell_U > 0 AND bessell_B > 0 and bessell_I > 0" #+
                            # "AND bessell_U_unc > 0 AND bessell_B_unc > 0 and bessell_V_unc > 0 and bessell_I_unc > 0"
                           ).format(rp['RAh range'][0], rp['RAh range'][1],rp['Dec range'][0],rp['Dec range'][1])

    #Filters
    rp['model_fnamelist'] = ['galex_NUV']+['bessell_{0}'.format(f) for f in ['U','B','V','I']]
    rp['data_fnamelist'] = ['bessell_{0}'.format(f) for f in ['U','B','V','I']]
    rp['fit_fnamelist'] = ['bessell_{0}'.format(f) for f in ['U','B','V','I']]

    # Root name for output files
    rp['outname'] = ('results/mcps_lmcPatch_{0}RAh{1}_{2}Dec{3}_V20_detect4'
                     .format(rp['RAh range'][0], rp['RAh range'][1],abs(rp['Dec range'][0]),abs(rp['Dec range'][1])))
    rp['outname'] = 'results/timing5'
    rp['logt_min'] = 3.4
    rp['logl_min'] = 1
    
elif rp['source'] is 'massey':
    rp['catalog_name'] = datadir + 'massey/massey02_lmc_table3a.dat'
    rp['spec_types'] = True
    rp['spec_catalog'] = datadir + 'massey/massey02_lmc_table4.dat'
    
    rp['model_fnamelist'] = ['galex_NUV']+['bessell_{0}'.format(f) for f in ['U','B','V','R']]
    rp['data_fnamelist'] = ['galex_NUV']+['bessell_{0}'.format(f) for f in ['U','B','V','R']]
    rp['fit_fnamelist'] = ['bessell_{0}'.format(f) for f in ['U','B','V','R']]

    rp['outname'] = 'results/massey_lmc_test'
    rp['logt_min'] = 3.7
    rp['logl_min'] = 2
    rp['galex_csv_filename'] = datadir + 'massey/massey_galex_lmc_sptyped.csv'

rp['dust_type'] = 'FM07' #FM07|LMC|SMC|F99.  In the latter cases, R_V should be set to None

# Parameters for which to return CDF moments
rp['outparnames'] = ['galex_NUV', 'LOGL', 'LOGT', 'A_V']#'R_V','F_BUMP', 'LOGM'] 
rp['return_residuals'] = True # False or True #return residuals from the best fit in each band


#### DO THE FITTING
# Initialize the dust attenuation curve and the stellar SED fitter
dust = extinction.Attenuator(dust_type = rp['dust_type'])
fitter = sf.StarfitterMCMC(rp)

fitter.lnp_prior = mcmodel.priors_from_isochrone
fitter.model = mcmodel.model
fitter.lnprob = mcmodel.lnprob

fitter.load_data()
# Change data to fit
#fitter.rp['nlines'] = 1e4
#fitter.load_data()
#fitter.setup_output

# Fit all objexts/pixels
fitter.fit_image()

#Write results to a fits binary table
catio.write_fitsbinary(fitter, outparlist = [ 'galex_NUV','LOGT','LOGL', 'A_V'])

####### PLOTTING #########

if rp['source'] is 'massey':
    plotter.plot_sptypes(fitter)
    pl.xlim(9,42)
    pl.savefig(rp['outname']+'_sptype_byclass.png')
    plotter.plot_sptypes(fitter, cpar = 'A_V')
    pl.xlim(9,42)
    pl.savefig(rp['outname']+'_sptype_byAV.png')

    if 'galex_NUV' in rp['fnamelist']:
        plotter.plot_sptypes(fitter, cpar = fitter.rp['data_header']['fov_radius'])
        pl.xlim(9,42)
        pl.savefig(rp['outname']+'_sptype_byFOVradius.png')
        pl.xlim(9,42)
        plotter.plot_sptypes(fitter, cpar = fitter.rp['data_header']['NUV_exptime'])
        pl.xlim(9,42)
        pl.savefig(rp['outname']+'_sptype_byNUVexptime.png')


plot_switch = False
if plot_switch:
    gf = {'goodfit':(fitter.rp['data_header']['spType'] != '') & (fitter.max_lnprob[0,:]*(-2) < 100),
          'glabel': r'spTyped $\chi^2 < 100$'}
      #gf = {'goodfit':(np.char.find(fitter.rp['data_header']['spType'],'WN') >=0) & (fitter.max_lnprob[0,:]*(-2) < 100),
      #'glabel': r'WN $\chi^2 < 100$'}
    plotter.plot_precision(fitter, PAR = 'LOGT', **gf)
    #pl.savefig('logt_unc.png')
    #plotter.plot_precision(fitter, PAR = 'A_V', **gf)
    #plotter.plot_precision(fitter, PAR = 'A_V',versus = fitter.parval['LOGT'][0,:,1], **gf)
    plotter.plot_precision(fitter, PAR = 'galex_NUV', **gf)
    plotter.plot_pars(fitter, PAR1 = 'LOGT', PAR2 = 'LOGL', loc = 4, **gf)
    pl.savefig(rp['outname']+'_logl_logt.png')
    plotter.plot_pars(fitter, PAR1 = 'LOGT', PAR2 = 'A_V', **gf)
    plotter.residuals(fitter, bands = [0, 1, 2, 3], colors  = ['m','b','g','r'], **gf)

#### Plot grid information

    pl.scatter(fitter.stargrid.pars['LOGT'],
               fitter.stargrid.sed[:,0]-fitter.stargrid.sed[:,4],
               c = fitter.stargrid.pars['A_V'], alpha = 0.5)
    pl.scatter(fitter.stargrid.sed[:,0]-fitter.stargrid.sed[:,1],
               fitter.stargrid.sed[:,1]-fitter.stargrid.sed[:,4],
               c = fitter.stargrid.pars['F_BUMP'], alpha = 0.5)
    pl.colorbar()
    pl.xlabel('NUV-U')
    pl.ylabel('U-R')
    pl.title('sedgrid by F_BUMP')
    pl.figure()
    pl.scatter(fitter.stargrid.sed[:,1]-fitter.stargrid.sed[:,2],
               fitter.stargrid.sed[:,2]-fitter.stargrid.sed[:,4],
               c = fitter.stargrid.pars['A_V'], alpha = 0.5)
    pl.colorbar()
