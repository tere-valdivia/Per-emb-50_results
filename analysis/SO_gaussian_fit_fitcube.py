'''
Author: Teresa Valdivia-Mena
Last revised August 31, 2022

This code is to fit 1, 2, and 3 Gaussian components to a small SO (55-44)
cube centered around Per-emb-50. Then, the routine determines for each spectrum
which is the best fit using the Akaike Informtion Criterion.


'''

import numpy as np
from scipy import stats
from spectral_cube import SpectralCube
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.modeling.functional_models import Gaussian1D
import os
import sys
sys.path.append('../')
from NOEMAsetup import *
import pyspeckit
import regions
import matplotlib.pyplot as plt


def chi_square(yPred, yData, err):
    chi2 = np.sum((yPred-yData)**2/(err**2))
    return chi2

# We use the AIC definition used in Choudhury et al 2020 (with least square
# instead of the maximum likelihood)
def AIC(yPred, data, err, k):
    """
    Returns the Akaike information criterion (AIC) for a given function with a
    number of parameters k and a negative log-likelihood value given
    by func(data, params)
    """
    # ll = log_likelihood(yPred, data, err)
    # aic = 2 * k + 2 * ll
    chi2 = chi_square(yPred, data, err)
    aic = 2 * k + chi2 # we leave out the constant because it is the same for
    # both models
    return aic

def probaic(aicmin, aiclist):
    return np.exp((aicmin-aiclist)/2)


def filter3G(spc, rms, rmslevel, errorfrac=0.5, epsilon=1.e-5, negative=True, velmin=None, velmax=None):
    """
    Replace the pixels in the fitted cube with np.nan where the fit is not
    good enough according to our criteria.

    The criteria that a pixel must have are:
    - The error is not zero
    - The value for each peak must not be negative (in this case we know the
    moment 1 must be positive, so we specify negative=True, can be changed)
    - If one pixel in a spectra is np.nan, all the spectra must be nan (sanity
    check)
    - The error fraction is lower than errorfrac
    - The peak value must be larger than rms times rmslevel
    - If defined, central velocity must be between velmin and velmax
    Args:
        variable (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """
    # error different than 0
    zeromask = np.where(np.abs(spc.errcube[0]) < epsilon, 1, 0) + \
        np.where(np.abs(spc.errcube[1]) < epsilon, 1, 0) + \
        np.where(np.abs(spc.errcube[2]) < epsilon, 1, 0) + \
        np.where(np.abs(spc.errcube[3]) < epsilon, 1, 0) + \
        np.where(np.abs(spc.errcube[4]) < epsilon, 1, 0) + \
        np.where(np.abs(spc.errcube[5]) < epsilon, 1, 0) + \
        np.where(np.abs(spc.errcube[6]) < epsilon, 1, 0) + \
        np.where(np.abs(spc.errcube[7]) < epsilon, 1, 0) + \
        np.where(np.abs(spc.errcube[8]) < epsilon, 1, 0)
    spc.parcube[np.where(np.repeat([zeromask], 9, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([zeromask], 9, axis=0))] = np.nan
    if negative:
        negativemask = np.where(spc.parcube[0] < 0, 1, 0) + \
            np.where(spc.parcube[1] < 0, 1, 0) + \
            np.where(spc.parcube[2] < 0, 1, 0)+ \
            np.where(spc.parcube[3] < 0, 1, 0)+ \
            np.where(spc.parcube[4] < 0, 1, 0)+ \
            np.where(spc.parcube[5] < 0, 1, 0)+ \
            np.where(spc.parcube[6] < 0, 1, 0)+ \
            np.where(spc.parcube[7] < 0, 1, 0)+ \
            np.where(spc.parcube[8] < 0, 1, 0)
        spc.parcube[np.where(np.repeat([negativemask], 9, axis=0))] = np.nan
        spc.errcube[np.where(np.repeat([negativemask], 9, axis=0))] = np.nan
    errormask = np.where(np.abs(spc.errcube[0]/spc.parcube[0]) > errorfrac, 1, 0)\
        + np.where(np.abs(spc.errcube[1]/spc.parcube[1]) > errorfrac, 1, 0)\
        + np.where(np.abs(spc.errcube[2]/spc.parcube[2]) > errorfrac, 1, 0)\
        + np.where(np.abs(spc.errcube[3]/spc.parcube[3]) > errorfrac, 1, 0)\
        + np.where(np.abs(spc.errcube[4]/spc.parcube[4]) > errorfrac, 1, 0)\
        + np.where(np.abs(spc.errcube[5]/spc.parcube[5]) > errorfrac, 1, 0)\
        + np.where(np.abs(spc.errcube[6]/spc.parcube[6]) > errorfrac, 1, 0)\
        + np.where(np.abs(spc.errcube[7]/spc.parcube[7]) > errorfrac, 1, 0)\
        + np.where(np.abs(spc.errcube[8]/spc.parcube[8]) > errorfrac, 1, 0)
    spc.parcube[np.where(np.repeat([errormask], 9, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([errormask], 9, axis=0))] = np.nan
    nanmask = np.sum(np.where(np.isnan(np.concatenate([spc.parcube, spc.errcube])), 1, 0), axis=0)
    spc.parcube[np.where(np.repeat([nanmask], 9, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([nanmask], 9, axis=0))] = np.nan
    peakmask = np.where(spc.parcube[0] < rmslevel*rms, 1, 0) + \
                np.where(spc.parcube[3] < rmslevel*rms, 1, 0) + \
                np.where(spc.parcube[6] < rmslevel*rms, 1, 0)
    spc.parcube[np.where(np.repeat([peakmask], 9, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([peakmask], 9, axis=0))] = np.nan
    # sigmamask =
    if velmin is not None:
        vel1mask = np.where(spc.parcube[1] < velmin, 1, 0) + \
            np.where(spc.parcube[4] < velmin, 1, 0)+ \
            np.where(spc.parcube[7] < velmin, 1, 0)
        spc.parcube[np.where(np.repeat([vel1mask], 9, axis=0))] = np.nan
        spc.errcube[np.where(np.repeat([vel1mask], 9, axis=0))] = np.nan
    if velmax is not None:
        vel2mask = np.where(spc.parcube[1] > velmax, 1, 0) + \
            np.where(spc.parcube[4] > velmax, 1, 0)+ \
            np.where(spc.parcube[7] > velmax, 1, 0)
        spc.parcube[np.where(np.repeat([vel2mask], 9, axis=0))] = np.nan
        spc.errcube[np.where(np.repeat([vel2mask], 9, axis=0))] = np.nan
    return spc

def filter2G(spc, rms, rmslevel, errorfrac=0.5, epsilon=1.e-5, negative=True, velmin=None, velmax=None):
    """
    Replace the pixels in the fitted cube with np.nan where the fit is not
    good enough according to our criteria.

    The criteria that a pixel must have are:
    - The error is not zero
    - The value for each parameter must not be negative (in this case we know the
    moment 1 must be positive, so we specify negative=True, can be changed)
    - If one pixel in a spectra is np.nan, all the spectra must be nan (sanity
    check)
    - The error fraction is lower than errorfrac
    - If defined, central velocity must be between velmin and velmax
    Args:
        variable (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """
    # error different than 0
    zeromask = np.where(np.abs(spc.errcube[0]) < epsilon, 1, 0) + \
        np.where(np.abs(spc.errcube[1]) < epsilon, 1, 0) + \
        np.where(np.abs(spc.errcube[2]) < epsilon, 1, 0) + \
        np.where(np.abs(spc.errcube[3]) < epsilon, 1, 0) + \
        np.where(np.abs(spc.errcube[4]) < epsilon, 1, 0) + \
        np.where(np.abs(spc.errcube[5]) < epsilon, 1, 0)
    spc.parcube[np.where(np.repeat([zeromask], 6, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([zeromask], 6, axis=0))] = np.nan
    if negative:
        negativemask = np.where(spc.parcube[0] < 0, 1, 0) + \
            np.where(spc.parcube[1] < 0, 1, 0) + \
            np.where(spc.parcube[2] < 0, 1, 0)+ \
            np.where(spc.parcube[3] < 0, 1, 0)+ \
            np.where(spc.parcube[4] < 0, 1, 0)+ \
            np.where(spc.parcube[5] < 0, 1, 0)
        spc.parcube[np.where(np.repeat([negativemask], 6, axis=0))] = np.nan
        spc.errcube[np.where(np.repeat([negativemask], 6, axis=0))] = np.nan
    errormask = np.where(np.abs(spc.errcube[0]/spc.parcube[0]) > errorfrac, 1, 0)\
        + np.where(np.abs(spc.errcube[1]/spc.parcube[1]) > errorfrac, 1, 0)\
        + np.where(np.abs(spc.errcube[2]/spc.parcube[2]) > errorfrac, 1, 0)\
        + np.where(np.abs(spc.errcube[3]/spc.parcube[3]) > errorfrac, 1, 0)\
        + np.where(np.abs(spc.errcube[4]/spc.parcube[4]) > errorfrac, 1, 0)\
        + np.where(np.abs(spc.errcube[5]/spc.parcube[5]) > errorfrac, 1, 0)
    spc.parcube[np.where(np.repeat([errormask], 6, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([errormask], 6, axis=0))] = np.nan
    nanmask = np.sum(np.where(np.isnan(np.concatenate([spc.parcube, spc.errcube])), 1, 0), axis=0)
    spc.parcube[np.where(np.repeat([nanmask], 6, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([nanmask], 6, axis=0))] = np.nan
    peakmask = np.where(spc.parcube[0] < rmslevel*rms, 1, 0) + \
                np.where(spc.parcube[3] < rmslevel*rms, 1, 0)
    spc.parcube[np.where(np.repeat([peakmask], 6, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([peakmask], 6, axis=0))] = np.nan
    if velmin is not None:
        vel1mask = np.where(spc.parcube[1] < velmin, 1, 0) + \
            np.where(spc.parcube[4] < velmin, 1, 0)
        spc.parcube[np.where(np.repeat([vel1mask], 6, axis=0))] = np.nan
        spc.errcube[np.where(np.repeat([vel1mask], 6, axis=0))] = np.nan
    if velmax is not None:
        vel2mask = np.where(spc.parcube[1] > velmax, 1, 0) + \
            np.where(spc.parcube[4] > velmax, 1, 0)
        spc.parcube[np.where(np.repeat([vel2mask], 6, axis=0))] = np.nan
        spc.errcube[np.where(np.repeat([vel2mask], 6, axis=0))] = np.nan
    return spc

def filter1G(spc, rms, rmslevel, errorfrac=0.5, epsilon=1.e-5,negative=True, velmin=None, velmax=None):
    """
    Replace the pixels in the fitted cube with np.nan where the fit is not
    good enough according to our criteria.

    The criteria that a pixel must have are:
    - The error is not zero
    - The error fraction is lower than errorfrac
    - If one pixel in a spectra is np.nan, all the spectra must be nan (sanity
    check)

    Args:
        variable (type): description

    Returns:
        type:

    """
    zeromask = np.where(np.abs(spc.errcube[0]) < epsilon, 1, 0) + \
        np.where(np.abs(spc.errcube[1]) < epsilon, 1, 0) + \
        np.where(np.abs(spc.errcube[2]) < epsilon, 1, 0)
    spc.parcube[np.where(np.repeat([zeromask], 3, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([zeromask], 3, axis=0))] = np.nan
    if negative:
        negativemask = np.where(spc.parcube[0] < 0, 1, 0) + \
            np.where(spc.parcube[1] < 0, 1, 0) + \
            np.where(spc.parcube[2] < 0, 1, 0)
        spc.parcube[np.where(np.repeat([negativemask], 3, axis=0))] = np.nan
        spc.errcube[np.where(np.repeat([negativemask], 3, axis=0))] = np.nan
    errormask = np.where(np.abs(spc.errcube[0]/spc.parcube[0]) > errorfrac, 1, 0)\
        + np.where(np.abs(spc.errcube[1]/spc.parcube[1]) > errorfrac, 1, 0)\
        + np.where(np.abs(spc.errcube[1]/spc.parcube[1]) > errorfrac, 1, 0)
    spc.parcube[np.where(np.repeat([errormask], 3, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([errormask], 3, axis=0))] = np.nan
    # Force if one pixel in a channel is nan, all the same pixels
    # in all  channels must be nan
    nanmask = np.sum(np.where(np.isnan(np.concatenate([spc.parcube, spc.errcube])), 1, 0), axis=0)
    spc.parcube[np.where(np.repeat([nanmask], 3, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([nanmask], 3, axis=0))] = np.nan
    peakmask = np.where(spc.parcube[0] < rmslevel*rms, 1, 0)
    spc.parcube[np.where(np.repeat([peakmask], 3, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([peakmask], 3, axis=0))] = np.nan
    if velmin is not None:
        vel1mask = np.where(spc.parcube[1] < velmin, 1, 0)
        spc.parcube[np.where(np.repeat([vel1mask], 3, axis=0))] = np.nan
        spc.errcube[np.where(np.repeat([vel1mask], 3, axis=0))] = np.nan
    if velmax is not None:
        vel2mask = np.where(spc.parcube[1] > velmax, 1, 0)
        spc.parcube[np.where(np.repeat([vel2mask], 3, axis=0))] = np.nan
        spc.errcube[np.where(np.repeat([vel2mask], 3, axis=0))] = np.nan
    return spc


'''
We use the following assumption:
 "If the model residuals are expected to be normally distributed then a
 log-likelihood function based on the one above can be used." In R notation
 this is for a linear regression:
 LL <- function(beta0, beta1, mu, sigma) {
    R = y - x * beta1 - beta0
    #
    R = suppressWarnings(dnorm(R, mu, sigma, log = TRUE))
    #
    -sum(R)
}
'''
########
# Inputs
########
# cubefile = '../SO_55_44/CDconfigsmall/gaussian_fit_123G_fitcube2g/Per-emb-50_CD_l009l048_uvsub_SO_multi_small'
cubefile = '../' + SO_55_44_s
fitregionfile = 'SO_2G_fitregion.reg'
fitfolder = 'SO_fit_results/'
saveplots = False
velinit = -1.0 * u.km/u.s
velend = 13.0 * u.km/u.s
velmin = 5.0
velmax = 10.0
starting_point = (37, 24)
prob_tolerance = 0.05
snratio = 4 # minimal signal to noise ratio for the fitted amplitude
signal_cut = 4 # minimal signal to noise the data must have to fit
min_sigma = 0. # FWHM min 3 channels of 0.08 km/s each
########
# End Inputs
########

if not os.path.exists(cubefile+'_fitcube2g.fits'):
    # The cube to fit must be smaller than the small cube we set earlier
    # and must be in K and km/s
    cube1 = SpectralCube.read(cubefile+'.fits').with_spectral_unit(u.km/u.s)
    header1 = cube1.header
    restfreq = header1['restfreq'] * u.Hz
    beamarea = 1.133 * header1['bmaj'] * u.deg * header1['bmin'] * u.deg
    regionlist = regions.read_ds9(fitregionfile)
    subcube = cube1.subcube_from_regions(regionlist)
    subcube.hdu.writeto(cubefile+'_fitcube2g.fits')

cube = SpectralCube.read(cubefile+'_fitcube2g.fits')
cube = cube.to(u.K)
if not os.path.exists(cubefile+'_fitcube2g_K.fits'):
    cube.hdu.writeto(cubefile+'_fitcube2g_K.fits')
velaxis = cube.with_spectral_unit(u.km/u.s).spectral_axis
n_y, n_x = np.shape(cube.unmasked_data[0,:,:])
spc = pyspeckit.Cube(cube=cube)
header = cube.header
freq = (header['RESTFREQ']/1e9) * u.GHz
wcsspec = WCS(header).spectral
wcscel = WCS(header).celestial

# measure the noise only where there is no line emission
chanlims = [wcsspec.world_to_pixel(velinit).tolist(), wcsspec.world_to_pixel(velend).tolist()]
rms = np.nanstd(np.vstack([cube[:int(np.min(chanlims))], cube[int(np.max(chanlims)):]]))
rmsmap = np.ones(np.shape(spc.cube)) * rms
spc.errorcube = rmsmap

momentsfile = folder + cubefile + '_fitcube2g_moments.fits'
cube_mom = cube.spectral_slab(velinit,velend)
spc_mom = pyspeckit.Cube(cube=cube_mom)
if os.path.exists(momentsfile):
    spc.momentcube = fits.getdata(momentsfile)
else:
    spc_mom.momenteach(vheight=False)
    spc.momentcube = spc_mom.momentcube
    moments = fits.PrimaryHDU(data=spc.momentcube, header=wcscel.to_header())
    moments.writeto(momentsfile)
spc2 = spc.copy()
spc3 = spc.copy()

"""
1 gaussian
"""
mom01 = np.where(spc.momentcube[0]>rms*snratio, spc.momentcube[0],rms*snratio)
initguesses = np.array([mom01,spc.momentcube[1],spc.momentcube[2]])
fitfile1 = fitfolder + cubefile + '_1G_fitparams.fits'
if not os.path.exists(fitfile1):
    try:
        spc.fiteach(fittype='gaussian',
                    guesses=initguesses,
                    verbose_level=0,
                    verbose=0,
                    signal_cut=signal_cut,
                    blank_value=np.nan,
                    start_from_point=(starting_point))
    except AssertionError:
        print('There are non-finite parameters in the fit')
    spc.write_fit(fitfile1)
else:
    spc.load_model_fit(fitfile1, 3, fittype='gaussian')

'''
The pixels we sample are:

25,34 : 3:29:07.7919, +31:21:57.586 (northern rotation)
29, 25 : 3:29:07.7441, +31:21:56.211 (southern rotation)
37, 23 : 3:29:07.6487, +31:21:55.906 (northern tip of the streamer)
41, 13 : 3:29:07.6010, +31:21:54.378 (southern tip of the streamer)
40, 18 : 3:29:07.6097, +31:21:55.042 (border to where the stream becomes 1G)
30, 12 : 3:29:07.7322, +31:21:54.225 (just below the rotation)
33, 41 : 3:29:07.6964, +31:21:58.656 (northern inversion)
34, 24 : (for some reason,here I see two but the aic is telling one component)
28, 38 : one of the 3G problematic fits
'''
plot_x = [25,23,29,37,41,30,33,40,34,28,31]
plot_y = [34,36,25,23,13,12,41,18,24,38,35]

if saveplots:
    for x, y in zip(plot_x,plot_y):
        sp = spc.get_spectrum(x,y)
        sp.plotter()
        try:
            sp.specfit.plot_fit()
        except ValueError:
            print("{0}, {1} has no fit for 2G".format(x, y))
        sp.plotter.savefig(cubefile+ '_1G_'+str(x)+'_'+str(y)+'.pdf')
        plt.close('all')


"""
2 gaussians
"""

mom11 = np.ones(np.shape(spc.momentcube[1])) * 9.2
# mom11 = spc.momentcube[1] - 0.2
mom12 = np.ones(np.shape(spc.momentcube[1])) * 7.5
# mom12 = spc.momentcube[1] + 0.2
# mom21 = mom22 = np.where(np.sqrt(spc.momentcube[2])/2 > min_sigma, np.sqrt(spc.momentcube[2])/2, min_sigma)
mom21 = mom22 = np.ones(np.shape(spc.momentcube[2])) * 0.3
mom02 = mom01 # then will be divided by 2
initguesses2 = np.concatenate([[mom01/2, mom11, mom21], [mom02/2, mom12, mom22]])
if not os.path.exists(fitfolder + cubefile + '_2G_initguesses.fits'):
    initguesses2file = fits.PrimaryHDU(data=initguesses2)
    initguesses2file.writeto(fitfolder + cubefile + '_2G_initguesses.fits')

fitfile2 = fitfolder + cubefile + '_2G_fitparams.fits'
if os.path.exists(fitfile2):
    spc2.load_model_fit(fitfile2, 3, npeaks=2,fittype='gaussian')
else:
    try:
        spc2.fiteach(fittype='gaussian',
                    guesses=initguesses2,
                    negamp=False,
                    # parlimited=[(True,False), (False,False), (True,True),(True,False), (False,False), (True,True)],
                    # parlimits=[(rms*snratio,np.inf),(0,np.inf),(min_sigma,2.),(rms*snratio,np.inf),(0,np.inf),(min_sigma,2.)],
                    verbose_level=0,
                    verbose=0,
                    signal_cut=signal_cut,
                    blank_value=np.nan,
                    start_from_point=(starting_point))
    except AssertionError:
        print('There are non-finite parameters in the fit')
    # as there are some nan parameters, which indicate failed fits, this will
    # throw an error
    spc2.write_fit(fitfile2)

if saveplots:
    for x, y in zip(plot_x,plot_y):
        sp2 = spc2.get_spectrum(x,y)
        sp2.plotter()
        xarr = sp2.xarr.value
        params = spc2.parcube[:,y,x]
        try:
            sp2.specfit.plot_fit()
            g1 = Gaussian1D(amplitude=params[0],mean=params[1],stddev=params[2])
            g2 = Gaussian1D(amplitude=params[3],mean=params[4],stddev=params[5])
            plt.plot(xarr, g1(xarr), '--g')
            plt.plot(xarr, g2(xarr), '--b')
        except ValueError:
            print("{0}, {1} has no fit for 2G".format(x, y))
        sp2.plotter.savefig(cubefile+ '_2G_'+str(x)+'_'+str(y)+'.pdf')
        plt.close('all')

"""
3 gaussians (we allow the third gaussian component to be broader, but not
larger than 5 kms, as a first test)

"""
mom03 = mom01
# mom13 = mom11 - 0.2
mom13 = np.ones(np.shape(spc.momentcube[1])) * 6.8
mom23 = np.ones(np.shape(spc.momentcube[2])) * 2

initguesses3 = np.concatenate([[mom01, mom11, mom21], [mom02, mom12, mom22], [mom03, mom13, mom23]])
if not os.path.exists(fitfolder + cubefile + '_3G_initguesses.fits'):
    initguesses3file = fits.PrimaryHDU(data=initguesses3)
    initguesses3file.writeto(fitfolder + cubefile + '_3G_initguesses.fits')

fitfile3 = fitfolder + cubefile + '_3G_fitparams.fits'
if os.path.exists(fitfile3):
    spc3.load_model_fit(fitfile3, 3, npeaks=3,fittype='gaussian')
else:
    parlims = [(rms*snratio,np.inf),(0,np.inf),(min_sigma,2.),(rms*snratio,np.inf),(0,np.inf),(min_sigma,2.), (rms*snratio,np.inf),(0,np.inf),(min_sigma,5.)]
    try:
        spc3.fiteach(fittype='gaussian',
                    guesses=initguesses3,
                    negamp=False,
                    # parlimited=[(True,False), (False,False), (True,True),(True,False), (False,False), (True,True), (True,False), (False,False), (True,True)],
                    # parlimits=parlims,
                    verbose_level = 0,
                    verbose=0,
                    signal_cut=signal_cut,
                    blank_value=np.nan,
                    start_from_point=(starting_point))
    except AssertionError:
        print('There are non-finite parameters in the fit')
    # as there are some nan parameters, which indicate failed fits, this will
    # throw an error
    spc3.write_fit(fitfile3)

if saveplots:
    for x, y in zip(plot_x,plot_y):
        sp3 = spc3.get_spectrum(x,y)
        sp3.plotter()
        xarr = sp3.xarr.value
        params = spc3.parcube[:,y,x]
        try:
            sp3.specfit.plot_fit()
            g1 = Gaussian1D(amplitude=params[0],mean=params[1],stddev=params[2])
            g2 = Gaussian1D(amplitude=params[3],mean=params[4],stddev=params[5])
            g3 = Gaussian1D(amplitude=params[6],mean=params[7],stddev=params[8])
            plt.plot(xarr, g1(xarr), '--g')
            plt.plot(xarr, g2(xarr), '--b')
            plt.plot(xarr, g3(xarr), '--m')
        except ValueError:
            print("{0}, {1} has no fit for 3G".format(x, y))
        sp3.plotter.savefig(cubefile+ '_3G_'+str(x)+'_'+str(y)+'.pdf')
        plt.close('all')

# Apply the filters to the parameter cubes

spc = filter1G(spc, rms, snratio, velmin=velmin, velmax=velmax)
spc2 = filter2G(spc2, rms, snratio, velmin=velmin, velmax=velmax)
spc3 = filter3G(spc3, rms, snratio, velmin=velmin, velmax=velmax)

# save filtered map
newheadaic = wcscel.to_header()
fitfile1filtered = fitfolder + cubefile + '_1G_fitparams_filtered.fits'
if not os.path.exists(fitfile1filtered):
    spc.write_fit(fitfile1filtered)
fitfile2filtered = fitfolder + cubefile + '_2G_fitparams_filtered.fits'
if not os.path.exists(fitfile2filtered):
    spc2.write_fit(fitfile2filtered)
fitfile3filtered = fitfolder + cubefile + '_3G_fitparams_filtered.fits'
if not os.path.exists(fitfile3filtered):
    spc3.write_fit(fitfile3filtered)


fittedmodel1 = spc.get_modelcube()
fittedmodel2 = spc2.get_modelcube()
fittedmodel3 = spc3.get_modelcube()
# The AIC criterion is applied in the filtered cubes


aic1map = np.zeros(np.shape(mom12)) *np.nan
aic2map = np.zeros(np.shape(mom12))*np.nan
aic3map = np.zeros(np.shape(mom12))*np.nan
totalpix = n_x*n_y
flag_prob = np.zeros(np.shape(mom12))
for x in range(n_x):
    for y in range(n_y):
        # load the spectra and the parameters
        spectrum = cube[:, y, x]
        ypred1g = fittedmodel1[:, y, x]
        ypred2g = fittedmodel2[:, y, x]
        ypred3g = fittedmodel3[:, y, x]
        unit = spectrum.unit
        spectrum = spectrum.value
        params_1G = spc.parcube[:, y, x]
        params_2G = spc2.parcube[:, y, x]
        params_3G = spc3.parcube[:, y, x]
        # if the fit is nan in any of the pixels, ignore
        # if np.all(np.isnan(params_1G)) or np.all(np.isnan(params_2G)):
        #     # If one of the fits failed, the comparison does not make sense
        #     continue

        # evaluate the AIC for each model in the pixel
        aic1g = AIC(ypred1g, spectrum, rms, len(params_1G))
        aic1map[y,x] = aic1g
        aic2g = AIC(ypred2g, spectrum, rms, len(params_2G))
        aic2map[y,x] = aic2g
        aic3g = AIC(ypred3g, spectrum, rms, len(params_3G))
        aic3map[y,x] = aic3g
        # choose the minimum AIC
        aiclist = [aic1g, aic2g, aic3g]
        if np.all(np.isnan(aiclist)):
            continue
        print("Selecting best fit for pixel ({0},{1}) out of {2}".format(x,y,totalpix))
        minaicindex = np.nanargmin(aiclist)
        minaic = aiclist[minaicindex]

        if minaicindex + 1 == 1:
            # 1 gaussian fit is best
            spc2.parcube[:,y,x] = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
            spc2.errcube[:,y,x] = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
            spc3.parcube[:,y,x] = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
            spc3.errcube[:,y,x] = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]

        elif minaicindex + 1 == 2:
            # 2 gaussian fit is best
            spc.parcube[:,y,x] = [np.nan,np.nan,np.nan]
            spc.errcube[:,y,x] = [np.nan,np.nan,np.nan]

        else:
            # 3 gaussian fit is best
            spc.parcube[:,y,x] = [np.nan,np.nan,np.nan]
            spc.errcube[:,y,x] = [np.nan,np.nan,np.nan]
            spc2.parcube[:,y,x] = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
            spc2.errcube[:,y,x] = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
        aiclist.pop(minaicindex)
        prob = probaic(minaic, aiclist)
        if np.amax(prob) > prob_tolerance:
            flag_prob[y,x] = 1


fitfile1aicmap = fitfolder + cubefile + '_1G_fitparams_aicmap.fits'
if not os.path.exists(fitfile1aicmap):
    hduaic = fits.PrimaryHDU(data=aic1map, header=newheadaic)
    hduaic.writeto(fitfile1aicmap)
fitfile1aicres = fitfolder + cubefile + '_1G_fitparams_aicres.fits'
if not os.path.exists(fitfile1aicres):
    spc.write_fit(fitfile1aicres)

fitfile2aicmap = fitfolder + cubefile + '_2G_fitparams_aicmap.fits'
if not os.path.exists(fitfile2aicmap):
    hduaic = fits.PrimaryHDU(data=aic2map, header=newheadaic)
    hduaic.writeto(fitfile2aicmap)
fitfile2aicres = fitfolder + cubefile + '_2G_fitparams_aicres.fits'
if not os.path.exists(fitfile2aicres):
    spc2.write_fit(fitfile2aicres)

fitfile3aicmap = fitfolder + cubefile + '_3G_fitparams_aicmap.fits'
if not os.path.exists(fitfile3aicmap):
    hduaic = fits.PrimaryHDU(data=aic3map, header=newheadaic)
    hduaic.writeto(fitfile3aicmap)
fitfile3aicres = fitfolder + cubefile + '_3G_fitparams_aicres.fits'
if not os.path.exists(fitfile3aicres):
    spc3.write_fit(fitfile3aicres)

fitfileflags = fitfolder + cubefile + '_3G_flag.fits'
if not os.path.exists(fitfileflags):
    flaghdu = fits.PrimaryHDU(data=flag_prob, header=newheadaic)
    flaghdu.writeto(fitfileflags)
