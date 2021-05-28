import numpy as np
from scipy import stats
from spectral_cube import SpectralCube
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
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


def filter(spc, rms, rmslevel, negative=True, errorfrac=0.5, epsilon=1.e-5):
    """
    Replace the pixels in the fitted cube with np.nan where the fit is not
    good enough according to our criteria.
    The criteria that a pixel must have are:
    - The error is not zero
    - The value for each peak must not be negative (in this case we know the
    moment 1 must be positive, so we specify negative=True, can be changed)
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
    # no negative values
    if negative:
        negativemask = np.where(spc.parcube[0] < 0, 1, 0) + \
            np.where(spc.parcube[1] < 0, 1, 0) + \
            np.where(spc.parcube[2] < 0, 1, 0) + \
            np.where(spc.parcube[3] < 0, 1, 0) + \
            np.where(spc.parcube[4] < 0, 1, 0) + \
            np.where(spc.parcube[5] < 0, 1, 0)
        spc.parcube[np.where(np.repeat([negativemask], 6, axis=0))] = np.nan
        spc.errcube[np.where(np.repeat([negativemask], 6, axis=0))] = np.nan

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
cubefile = '../SO_55_44/CDconfigsmall/Per-emb-50_CD_l009l048_uvsub_SO_multi_small'
fitregionfile = 'SO_2G_fitregion.reg'
velinit = -1.0 * u.km/u.s
velend = 13.0 * u.km/u.s
starting_point = (37, 24)
prob_tolerance = 0.05
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

momentsfile = cubefile+'_fitcube2g_moments.fits'
cube_mom = cube.spectral_slab(velinit,velend)
spc_mom = pyspeckit.Cube(cube=cube_mom)
if os.path.exists(momentsfile):
    spc.momentcube = fits.getdata(momentsfile)
else:
    spc_mom.momenteach(vheight=False)
    spc.momentcube = spc_mom.momentcube
    moments = fits.PrimaryHDU(data=spc.momentcube)
    moments.writeto(momentsfile)
spc2 = spc.copy()
# We use the moments to do initial guesses
# Use the skew to determine the direction of the second gaussian

"""
1 gaussian
"""
initguesses = spc.momentcube

fitfile1 = cubefile + '_1G_fitparams.fits'
if not os.path.exists(fitfile1):
    spc.fiteach(fittype='gaussian',
                guesses=initguesses,
                use_neighbor_as_guess=True,
                verbose=1,
                errmap=rmsmap,
                signal_cut=4,
                blank_value=np.nan,
                start_from_point=(starting_point))
    spc.write_fit(fitfile1)
else:
    spc.load_model_fit(fitfile1, 3, fittype='gaussian')
#     fittedmodel = spc.get_modelcube()
#     spc.write_fit(fitfile, overwrite=True)
# spc = filter(spc, rms, 4)
# fitfile1masked = cubefile + '_1G_fitparams_masked.fits'
# if not os.path.exists(fitfile1masked):
#     spc.write_fit(fitfile1masked)
fittedmodel1 = spc.get_modelcube()

"""
2 gaussians
"""
mom11 = spc.momentcube[1] - 0.5
mom12 = spc.momentcube[1] + 0.5
mom21 = mom22 = np.sqrt(spc.momentcube[2])/2

mom01 = mom02 = np.where(spc.momentcube[0]>0, spc.momentcube[0],0)
# mom11 = spc.momentcube[1] - 0.2
# mom12 = spc.momentcube[1] + 0.2
initguesses2 = np.concatenate([[mom01, mom11, mom21], [mom02, mom12, mom22]])

# initguesses2file = fits.PrimaryHDU(data=initguesses2)
# initguesses2file.writeto(cubefile + '_2G_initguesses.fits')
fitfile2 = cubefile + '_2G_fitparams.fits'
if os.path.exists(fitfile2):
    spc2.load_model_fit(fitfile2, 3, npeaks=2,fittype='gaussian')
else:
    # TODO: force positive amplitude. This must be done spectra by spectra
    try:
        spc2.fiteach(fittype='gaussian',
                    guesses=initguesses2,
                    negamp=False,
                    # parlimited=[(True,False), (False,False), (False,False),(True,False), (False,False), (False,False)],
                    # parlimits=[(0,np.inf),(0,np.inf),(0,np.inf),(0,np.inf),(0,np.inf),(0,np.inf)],
                    verbose=3,
                    errmap=rmsmap,
                    signal_cut=4,
                    blank_value=np.nan,
                    start_from_point=(starting_point))
    except AssertionError:
        print('There are non-finite parameters in the fit')
    spc2.write_fit(fitfile2)

# spc2 = filter(spc2, rms, 4)
# fitfile2masked = cubefile + '_2G_fitparams_masked.fits'
# if not os.path.exists(fitfile2masked):
#     spc2.write_fit(fitfile2masked)
fittedmodel2 = spc2.get_modelcube()

# for now, we ignore the error in the parameters
# and also we do not filter with all criteria

# spc.parcube[:,25,25]
# spc2.parcube[:,20,31]
# sp = spc2.get_spectrum(25,25)
# sp.specfit(fittype='gaussian')
# sp.specfit.get_components()
# spc2.plot_spectrum(31,20, plot_fit=True)
# spc2.plot_spectrum(25,25, plot_fit=True)

aic1map = np.zeros(np.shape(mom12)) *np.nan
aic2map = np.zeros(np.shape(mom12))*np.nan
totalpix = n_x*n_y
flag_prob = np.zeros(np.shape(mom12))
for x in range(n_x):
    for y in range(n_y):
        # load the spectra and the parameters
        spectrum = cube[:, y, x]
        ypred1g = fittedmodel1[:, y, x]
        ypred2g = fittedmodel2[:, y, x]
        unit = spectrum.unit
        spectrum = spectrum.value
        params_1G = spc.parcube[:, y, x]
        params_2G = spc2.parcube[:, y, x]
        # if the fit is nan in any of the pixels, ignore
        # if np.all(np.isnan(params_1G)) or np.all(np.isnan(params_2G)):
        #     # If one of the fits failed, the comparison does not make sense
        #     continue
        print("Selecting best fit for pixel ({0},{1}) out of {2}".format(x,y,totalpix))
        # evaluate the AIC for each model in the pixel
        aic1g = AIC(ypred1g, spectrum, rms, len(params_1G))
        aic1map[y,x] = aic1g
        aic2g = AIC(ypred2g, spectrum, rms, len(params_2G))
        aic2map[y,x] = aic2g
        # choose the minimum AIC
        if aic2g < aic1g: #that 2G are best
            # for the model that is not the correct one, set the fit to NaN
            spc.parcube[:,y,x] = [np.nan,np.nan,np.nan]
            spc.errcube[:,y,x] = [np.nan,np.nan,np.nan]
            # we evaluate the probability that the other model is as good for
            # minimizing information loss as the best one
            prob = np.exp((aic2g - aic1g)/2.)
            if prob > prob_tolerance:
                flag_prob[y,x] = 1
        else:
            spc2.parcube[:,y,x] = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
            spc2.errcube[:,y,x] = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
            # prob = np.exp((aic1g - aic2g)/2.)
            # if prob > prob_tolerance:
            #     flag_prob[y,x] = 1

fitfile1aicmap = cubefile + '_1G_fitparams_aicmap.fits'
if not os.path.exists(fitfile1aicmap):
    hduaic = fits.PrimaryHDU(data=aic1map, header=wcscel.to_header())
    hduaic.writeto(fitfile1aicmap)
fitfile1aicres = cubefile + '_1G_fitparams_aicres.fits'
if not os.path.exists(fitfile1aicres):
    spc.write_fit(fitfile1aicres)

fitfile2aicmap = cubefile + '_2G_fitparams_aicmap.fits'
if not os.path.exists(fitfile2aicmap):
    hduaic = fits.PrimaryHDU(data=aic2map, header=wcscel.to_header())
    hduaic.writeto(fitfile2aicmap)
fitfile2aicres = cubefile + '_2G_fitparams_aicres.fits'
if not os.path.exists(fitfile2aicres):
    spc2.write_fit(fitfile2aicres)


fitfileflags = cubefile + '_2G_flag.fits'
if not os.path.exists(fitfileflags):
    flaghdu = fits.PrimaryHDU(data=flag_prob, header=wcscel.to_header())
    flaghdu.writeto(fitfileflags)



## see the spectra
