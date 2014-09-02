import sys
import numpy as np
from scipy.interpolate import interp1d

lsun, pc = 3.846e33, 3.085677581467192e18
lightspeed = 2.998e18 #AA/s
#value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2 )

def burst_sfh(fwhm_burst=0.05, f_burst=0.5, contrast=5,
              sfh=None, bin_res=10.):
    """
    Given a binned SFH as a numpy structured array, and burst
    parameters, generate a realization of the SFH at high temporal
    resolution. The output time resolution will be approximately
    fwhm_burst/12 unless no bursts are generated, in which case the
    output time resolution is the minimum bin width divided by
    bin_res.

    :param fwhm_burst: default 0.05
        the fwhm of the bursts to add, in Gyr.
        
    :param f_burst: default, 0.5
        the fraction of stellar mass formed in each bin that is formed
        in the bursts.
        
    :param contrast: default, 5
        the approximate maximum height or amplitude of the bursts
        above the constant background SFR.  This is only approximate
        since it is altered to preserve f_burst and fwhm_burst even
        though the number of busrsts is quantized.
        
    :param sfh: structured ndarray
        A binned sfh in numpy structured array format.  Usually the
        result of sfhutils.load_angst_sfh()
        
    :param bin_res: default 10
        Factor by which to increase the time resolution of the output
        grid, relative to the shortest bin width in the supplied SFH.

    :returns times:  ndarray of shape (nt)
        The output linear, regular temporal grid of lookback times.

    :returns sfr: ndarray of shape (nt)
        The resulting SFR at each time.

    :returns f_burst_actual:
        In case f_burst changed due to burst number discretezation.
        Shouldn't happen though.        
    """
    a, tburst, A, sigma, f_burst_actual = [],[],[],[],[]
    for i,abin in enumerate(sfh):
        res = convert_burst_pars(fwhm_burst = fwhm_burst, f_burst = f_burst, contrast = contrast,
                             bin_width = (abin['t2']-abin['t1']), bin_sfr = abin['sfr'])
        a += [res[0]]
        if len(res[1]) > 0:
            tburst += (res[1] + abin['t1']).tolist()
            A += len(res[1]) * [res[2]]
            sigma += len(res[1]) * [res[3]]
    if len(sigma) == 0:
        # If there were no bursts, set the time resolution to be
        # 1/bin_res of the shortest bin width.
        dt = (sfh['t2'] - sfh['t1']).min()/(1.0 * bin_res)
    else:
        dt = np.min(sigma)/5. #make sure you sample the bursts reasonably well
    times = np.arange(np.round(sfh['t2'].max()/dt)) * dt
    # Figure out which bin each time is in
    bins = [sfh[0]['t1']] + sfh['t2'].tolist()
    bin_num = np.digitize(times, bins) -1
    # Calculate SFR from all components
    sfr = np.array(a)[bin_num] + gauss(times, tburst, A, sigma)
    
    return times, sfr, f_burst_actual

def bursty_sps(lookback_time, lt, sfr, sps):
    """
    Obtain the spectrum of a stellar poluation with arbitrary complex
    SFH at a given lookback time.  The SFH is provided in terms of SFR
    vs t_lookback. Note that this in in contrast to the normal
    specification in terms of time since the big bang. Interpolation
    of the available SSPs to the time sequence of the SFH is
    accomplished by linear interpolation in log t.  Highly oscillatory
    SFHs require dense sampling of the temporal axis to obtain
    accurate results.

    :param lookback_time: scalar or ndarray, shape (ntarg)
        The lookback time(s) at which to obtain the spectrum. In yrs.
        
    :param lt: ndarray, shape (ntime)
        The lookback time sequence of the provided SFH.  Assumed to
        have have equal linear time intervals.
        
    :param sfr: ndarray, shape (ntime)
        The SFR corresponding to each element of lt, in M_sun/yr.
        
    :returns wave: ndarray, shape (nwave)
        The wavelength array
        
    :returns int_spec: ndarray, shape(ntarg, nwave)
        The integrated spectrum at lookback_time, in L_sun/AA
        
    :returns aw: ndarray, shape(ntarg, nage)
        The total weights of each SSP spectrum for each requested
        lookback_time.  Useful for debugging.        
    """
    
    dt = lt[1] - lt[0]
    sps.params['sfh'] = 0 #make sure using SSPs
    # Get *all* the ssps
    zmet = sps.params['zmet']-1
    spec, mass, _ = sps.all_ssp_spec(peraa =True, update = True)
    spec = spec[:,:,zmet].T
    wave = sps.wavelengths
    ssp_ages = 10**sps.ssp_ages #in yrs

    # Set up output
    target_lt = np.atleast_1d(lookback_time)
    int_spec = np.zeros( [ len(target_lt), len(wave) ] )
    mstar = np.zeros( len(target_lt) )
    aw = np.zeros( [ len(target_lt), len(ssp_ages) ] )

    for i,tlt in enumerate(target_lt):
        valid = (lt >= tlt) #only consider time points in the past of this lookback time.
        inds, weights = weights_1DLinear(np.log(ssp_ages), np.log(lt[valid] - tlt))
        # Aggregate the weights for each ssp time index, after accounting for SFR
        agg_weights = np.bincount( inds.flatten(),
                                   weights = (weights * sfr[valid,None]).flatten(),
                                   minlength = len(ssp_ages) ) * dt
        int_spec[i,:] = (spec * agg_weights[:,None]).sum(axis = 0)
        aw[i,:] = agg_weights
        mstar[i] = (mass[:,zmet] * agg_weights).sum()
    return wave, int_spec, aw, mstar


def bursty_lf(lookback_time, lt, sfr, sps_lf):
    """
    Obtain the luminosity function of stars for an arbitrary complex
    SFH at a given lookback time.  The SFH is provided in terms of SFR
    vs t_lookback. Note that this in in contrast to the normal
    specification in terms of time since the big bang.

    :param lookback_time: scalar or ndarray, shape (ntarg)
        The lookback time(s) at which to obtain the spectrum. In yrs.
        
    :param lt: ndarray, shape (ntime)
        The lookback time sequence of the provided SFH.  Assumed to
        have have equal linear time intervals.
        
    :param sfr: ndarray, shape (ntime)
        The SFR corresponding to each element of lt, in M_sun/yr.
        
    :param sps_lf:
        Luminosity function information, as a dictionary.  The keys of
        the dictionary are 'bins', 'lf' and 'ssp_ages'

    :returns bins:
        The bins used to define the LF
        
    :returns int_lf: ndarray, shape(ntarg, nbin)
        The integrated LF at lookback_time, in L_sun/AA
        
    :returns aw: ndarray, shape(ntarg, nage)
        The total weights of each LF for each requested
        lookback_time.  Useful for debugging.
        
    """
    dt = lt[1] - lt[0]
    bins, lf, ssp_ages = sps_lf['bins'], sps_lf['lf'], 10**sps_lf['ssp_ages']

    # Set-up output
    target_lt = np.atleast_1d(lookback_time)
    int_lf = np.zeros( [ len(target_lt), len(bins) ] )
    aw = np.zeros( [ len(target_lt), len(ssp_ages) ] )

    for i,tl in enumerate(target_lt):
        valid = (lt >= tl) #only consider time points in the past of this lookback time.
        inds, weights = weights_1DLinear(np.log(ssp_ages), np.log(lt[valid] - tl))
        #aggregate the weights for each ssp time index, after accounting for SFR
        agg_weights = np.bincount( inds.flatten(),
                                   weights = (weights * sfr[valid,None]).flatten(),
                                   minlength = len(ssp_ages) ) * dt
        int_lf[i,:] = (lf * agg_weights[:,None]).sum(axis = 0)
        aw[i,:] = agg_weights

    return bins, int_lf, aw


def gauss(x, mu, A, sigma):
    """
    Project the sum of a sequence of gaussians onto the x vector,
    using broadcasting.

    :param x: ndarray
        The array onto which the gaussians are to be projected.
        
    :param mu:
        Sequence of gaussian centers, same units as x.

    :param A:
        Sequence of gaussian normalization (that is, the area of the
        gaussians), same length as mu.
        
    :param sigma:
        Sequence of gaussian standard deviations or dispersions, same
        length as mu.

    :returns value:
       The value of the sum of the gaussians at positions x.
        
    """
    mu, A, sigma = np.atleast_2d(mu), np.atleast_2d(A), np.atleast_2d(sigma)
    val = A/(sigma * np.sqrt(np.pi * 2)) * np.exp(-(x[:,None] - mu)**2/(2 * sigma**2))
    return val.sum(axis = -1)


def convert_burst_pars(fwhm_burst = 0.05, f_burst=0.5, contrast=5,
                       bin_width=1.0, bin_sfr=1e9):

    """
    Perform the conversion from a burst fraction, width, and
    'contrast' to to a set of gaussian bursts stochastically
    distributed in time, each characterized by a burst time, a width,
    and an amplitude.  Also returns the SFR in the non-bursting mode.

    :param fwhm_burst: default 0.05
        The fwhm of the bursts to add, in Gyr.
        
    :param f_burst: default, 0.5
        The fraction of stellar mass formed in each bin that is formed
        in the bursts.
        
    :param contrast: default, 5
        The approximate maximum height or amplitude of the bursts
        above the constant background SFR.  This is only approximate
        since it is altered to preserve f_burst and fwhm_burst even
        though the number of busrsts is quantized.

    :param bin_width: default, 1.0
        The width of the bin in Gyr.

    :param bin_sfr:
        The average sfr for this time period.  The total stellar mass
        formed during this bin is just bin_sfr * bin_width.

    :returns a:
        The sfr of the non bursting constant component

    :returns tburst:
        A sequence of times, of length nburst, where the time gives
        the time of the peak of the gaussian burst
        
    :returns A:
        A sequence of normalizations of length nburst.  each A value
        gives the stellar mass formed in that burst.

    :returns sigma:
        A sequence of burst widths.  This is usually just
        fwhm_burst/2.35 repeated nburst times.
    """
    width, mstar = bin_width, bin_width * bin_sfr
    if width < fwhm_burst * 2:
        f_burst = 0.0 #no bursts if bin is short - they are resolved
    # Constant SF component
    a = mstar * (1 - f_burst) /width
    # Determine burst_parameters
    sigma = fwhm_burst / 2.355
    maxsfr = contrast * a
    A = maxsfr * (sigma * np.sqrt(np.pi * 2))
    tburst = []
    if A > 0:
        nburst = np.round(mstar * f_burst / A)
        # Recalculate A to preserve total mass formed in the face of
        # burst number quantization
        if nburst > 0:
            A = mstar * f_burst / nburst
            tburst = np.random.uniform(0,width, nburst)
        else:
            A = 0
            a = mstar/width
    else:
        nburst = 0
        a = mstar/width

    return [a, tburst, A, sigma]


def read_lfs(filename):
    """
    Read a Villaume/FSPS produced cumulative LF file, interpolate LFs
    at each age to a common magnitude grid, and return a dictionary
    containing the interpolated CLFs and ancillary information.

    :param filename:
        The filename (including path) of the Villaume CLF file

    :returns luminosity_func:
        A dictionary with the following key-value pairs:
        ssp_ages: Log of the age for each CLF, ndarray of shape (nage,)
        lf:       The interpolated CLFs, ndarray of shape (nage, nmag)
        bins:     Magnitude grid for the interpolated CLFs, ndarray of
                  shape (nmag,)
        orig:     2-element list contining the original magnitude grids
                  and CLFs as lists.
        
    """
    age, bins, lfs = [], [], []
    f = open(filename, "r")
    for i,line in enumerate(f):
        dat = [ float(d) for d in line.split() ]
        if (i % 3) == 0:
            age += [ dat[0]]
        elif (i % 3) == 1:
            bins += [dat]
        elif (i % 3) == 2:
            lfs += [dat]
    f.close()
    
    age = np.array(age)
    minage, maxage = np.min(age)-0.05, np.max(age)+0.10
    minl, maxl = np.min(bins)[0], np.max(bins)[0]+0.01
    allages = np.arange(minage, maxage, 0.05)
    mags = np.arange(minl, maxl, 0.01)
    print(minl, maxl)
    
    lf = np.zeros([ len(allages), len(mags)])
    for i, t in enumerate(allages):
        inds = np.isclose(t,age)
        if inds.sum() == 0:
            continue
        ind = np.where(inds)[0][0]
        x = np.array(bins[ind] + [np.max(mags)])
        y = np.log10(lfs[ind] +[np.max(lfs[ind])])        
        lf[i, :] = 10**interp1d(np.sort(x), np.sort(y), fill_value = -np.inf, bounds_error = False)(mags)

    luminosity_func ={}
    luminosity_func['ssp_ages'] = allages
    luminosity_func['lf'] = lf
    luminosity_func['bins'] = mags
    luminosity_func['orig'] = [bins, lfs]

    return luminosity_func

def weights_1DLinear(model_points, target_points, extrapolate = False):
    """
    The interpolation weights are determined from 1D linear
    interpolation.
    
    :param model_points: ndarray, shape(nmod)
        The parameter coordinate of the available models.  assumed to
        be sorted ascending
                
    :param target_points: ndarray, shape(ntarg)
        The coordinate to which you wish to interpolate
            
    :returns inds: ndarray, shape(ntarg,2)
         The model indices of the interpolates
             
    :returns weights: narray, shape (ntarg,2)
         The weights of each model given by ind in the interpolates.
    """
    #well this is ugly.
    mod_sorted = model_points
    
    x_new_indices = np.searchsorted(mod_sorted, target_points)
    x_new_indices = x_new_indices.clip(1, len(mod_sorted)-1).astype(int)
    lo = x_new_indices - 1
    hi = x_new_indices
    x_lo = mod_sorted[lo]
    x_hi = mod_sorted[hi]
    width = x_hi - x_lo    
    w_lo = (x_hi - target_points)/width
    w_hi = (target_points - x_lo)/width

    if extrapolate is False:
        #and of course I have these labels backwards
        above_scale = w_lo < 0 #fidn places where target is above or below the model range
        below_scale = w_hi < 0
        lo[above_scale] = hi[above_scale] #set the indices to be indentical in these cases
        hi[below_scale] = lo[below_scale]
        w_lo[above_scale] = 0 #make the combined weights sum to one
        w_hi[above_scale] = 1
        w_hi[below_scale] = 0
        w_lo[below_scale] = 1

    inds = np.vstack([lo,hi]).T
    weights = np.vstack([w_lo, w_hi]).T
    #inds = order[inds]
    return inds, weights
