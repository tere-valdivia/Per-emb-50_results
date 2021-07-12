import sys
sys.path.append('../')
from NOEMAsetup import *
from spectral_cube import SpectralCube
import pyspeckit
import matplotlib.pylab as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import regions
import os
from astropy.modeling.functional_models import Gaussian2D
from astropy.coordinates import SkyCoord

"""
This code is to fit one and two gaussian components in the cube and then choose
the best using the aic criteria
"""

def chi_square(yPred, yData, err):
    chi2 = np.sum((yPred-yData)**2/(err**2))
    return chi2

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


# def filter(spc, rms, rmslevel, velinit, velend, negative=True, errorfrac=0.5, epsilon=1.e-5):
#     (spc, rms, rmslevel, errorfrac=0.5, epsilon=1.e-5, negative=True)
#     """
#     Replace the pixels in the fitted cube with np.nan where the fit is not
#     good enough according to our criteria.
#
#     The criteria that a pixel must have are:
#     - The error is not zero
#     - The value for each peak must not be negative (in this case we know the
#     moment 1 must be positive, so we specify negative=True, can be changed)
#     - The error fraction is lower than errorfrac
#     - The moment 1 value must be within the range [velinit,velend]
#     - The peak value for both gaussian fits must be larger than rms times rmslevel
#     - The weighted velocity dispersion must be smaller than the absolute
#     value of velend-velinit
#     - If one pixel in a spectra is np.nan, all the spectra must be nan (sanity
#     check)
#
#     Args:
#         variable (type): description
#
#     Returns:
#         type: description
#
#     Raises:
#         Exception: description
#
#     """
#     # error different than 0
#     zeromask = np.where(np.abs(spc.errcube[0]) < epsilon, 1, 0) + \
#         np.where(np.abs(spc.errcube[1]) < epsilon, 1, 0) + \
#         np.where(np.abs(spc.errcube[2]) < epsilon, 1, 0) + \
#         np.where(np.abs(spc.errcube[3]) < epsilon, 1, 0) + \
#         np.where(np.abs(spc.errcube[4]) < epsilon, 1, 0) + \
#         np.where(np.abs(spc.errcube[5]) < epsilon, 1, 0)
#     spc.parcube[np.where(np.repeat([zeromask], 6, axis=0))] = np.nan
#     spc.errcube[np.where(np.repeat([zeromask], 6, axis=0))] = np.nan
#     # no negative values
#     if negative:
#         negativemask = np.where(spc.parcube[0] < 0, 1, 0) + \
#             np.where(spc.parcube[1] < 0, 1, 0) + \
#             np.where(spc.parcube[2] < 0, 1, 0) + \
#             np.where(spc.parcube[3] < 0, 1, 0) + \
#             np.where(spc.parcube[4] < 0, 1, 0) + \
#             np.where(spc.parcube[5] < 0, 1, 0)
#         spc.parcube[np.where(np.repeat([negativemask], 6, axis=0))] = np.nan
#         spc.errcube[np.where(np.repeat([negativemask], 6, axis=0))] = np.nan
#     # maximum acceptable error fraction
#     errormask = np.where(np.abs(spc.errcube[0]/spc.parcube[0]) > errorfrac, 1, 0)\
#         + np.where(np.abs(spc.errcube[1]/spc.parcube[1]) > errorfrac, 1, 0)\
#         + np.where(np.abs(spc.errcube[2]/spc.parcube[2]) > errorfrac, 1, 0)\
#         + np.where(np.abs(spc.errcube[3]/spc.parcube[3]) > errorfrac, 1, 0)\
#         + np.where(np.abs(spc.errcube[4]/spc.parcube[4]) > errorfrac, 1, 0)\
#         + np.where(np.abs(spc.errcube[5]/spc.parcube[5]) > errorfrac, 1, 0)
#     spc.parcube[np.where(np.repeat([errormask], 6, axis=0))] = np.nan
#     spc.errcube[np.where(np.repeat([errormask], 6, axis=0))] = np.nan
#     velocitymask = np.where(spc.parcube[1] < velinit.value, 1, 0) + \
#         np.where(spc.parcube[1] > velend.value, 1, 0) + \
#         np.where(spc.parcube[4] > velend.value, 1, 0) + \
#         np.where(spc.parcube[4] < velinit.value, 1, 0)
#     spc.parcube[np.where(np.repeat([velocitymask], 6, axis=0))] = np.nan
#     spc.errcube[np.where(np.repeat([velocitymask], 6, axis=0))] = np.nan
#     peakmask = np.where(spc.parcube[0] > np.amax(spc.cube), 1, 0) + \
#                 np.where(spc.parcube[3] > np.amax(spc.cube), 1, 0)
#     # peakmask = np.where(spc.parcube[0] < rmslevel*rms, 1, 0) + \
#     #     np.where(spc.parcube[3] < rmslevel*rms, 1, 0)
#     spc.parcube[np.where(np.repeat([peakmask], 6, axis=0))] = np.nan
#     spc.errcube[np.where(np.repeat([peakmask], 6, axis=0))] = np.nan
#     # sigmamask = np.where(spc.parcube[2] > (velend-velinit).value/2, 1, 0) + \
#     #     np.where(spc.parcube[2] < np.abs(header['cdelt3']), 1, 0) + \
#     #     np.where(spc.parcube[5] > (velend-velinit).value/2, 1, 0) + \
#     #     np.where(spc.parcube[5] < np.abs(header['cdelt3']), 1, 0)
#     # spc.parcube[np.where(np.repeat([sigmamask], 6, axis=0))] = np.nan
#     # spc.errcube[np.where(np.repeat([sigmamask], 6, axis=0))] = np.nan
#     # Force if one pixel in a channel is nan, all the same pixels
#     # in all  channels must be nan
#     # nanmask = np.sum(np.where(np.isnan(np.concatenate([spc.parcube, spc.errcube])), 1, 0), axis=0)
#     # spc.parcube[np.where(np.repeat([nanmask], 6, axis=0))] = np.nan
#     # spc.errcube[np.where(np.repeat([nanmask], 6, axis=0))] = np.nan
#     return spc


def filter2G(spc, rms, rmslevel, errorfrac=0.5, epsilon=1.e-5, negative=True):
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
    # peakmask = np.where(spc.parcube[0] < rmslevel*rms, 1, 0) + \
    #             np.where(spc.parcube[3] < rmslevel*rms, 1, 0)
    # spc.parcube[np.where(np.repeat([peakmask], 6, axis=0))] = np.nan
    # spc.errcube[np.where(np.repeat([peakmask], 6, axis=0))] = np.nan
    return spc

def filter1G(spc, rms, rmslevel, errorfrac=0.5, epsilon=1.e-5,negative=True):
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
    # peakmask = np.where(spc.parcube[0] < rmslevel*rms, 1, 0)
    # spc.parcube[np.where(np.repeat([peakmask], 3, axis=0))] = np.nan
    # spc.errcube[np.where(np.repeat([peakmask], 3, axis=0))] = np.nan
    return spc


pbcor = False
cubefile = '../SO2_11_1_11_10_0_10/CDconfig/Per-emb-50_CD_l031l070_uvsub_SO2_multi'
signal_cut = 3
snratio = 3
prob_tolerance = 0.05
# cubefile_nonpb = '../' + H2CO_303_202_s
# cubefile = '../C18O/CDconfig/JEP/JEP_mask_multi_Per-emb-50_CD_l025l064_uvsub_C18O_pbcor'
# cubefile_nonpb = '../C18O/CDconfig/JEP/JEP_mask_multi_Per-emb-50_CD_l025l064_uvsub_C18O'
# Where we estimate the line is
# for SO2
# cubefile = '../SO_55_44/CDconfigsmall/Per-emb-50_CD_l009l048_uvsub_SO_multi_small'
velinit = 5.5 * u.km/u.s
velend = 12.0 * u.km/u.s
# for C18O
# velinit = 5.5 * u.km/u.s
# velend = 9.5 * u.km/u.s
# cubefile = '../SO_55_44/CDconfigsmall/Per-emb-50_CD_l009l048_uvsub_SO_multi_small'
# fitregionfile = '../analysis/H2CO_fitregion.reg'
fitregionfile = 'SO_2G_fitregion.reg'
# starting_point = (70, 82) #H2CO
# starting_point = (34, 27) # SO
starting_point = (28, 28)  #SO2
# starting_point = (126,135) #C18O

if not os.path.exists(cubefile+'_fitcube2g.fits'):
    # The cube to fit must be smaller than the small cube we set earlier
    # and must be in K and km/s
    cube1 = SpectralCube.read(cubefile+'.fits').with_spectral_unit(u.km/u.s)
    header1 = cube1.header
    restfreq = header1['restfreq'] * u.Hz
    beamarea = 1.133 * header1['bmaj'] * u.deg * header1['bmin'] * u.deg
    cube1 = cube1.to(u.K, u.brightness_temperature(restfreq, beam_area=beamarea))
    regionlist = regions.read_ds9(fitregionfile)
    subcube = cube1.subcube_from_regions(regionlist)
    subcube.hdu.writeto(cubefile+'_fitcube2g.fits')


cube = SpectralCube.read(cubefile+'_fitcube2g.fits')
velaxis = cube.with_spectral_unit(u.km/u.s).spectral_axis
n_y, n_x = np.shape(cube.unmasked_data[0,:,:])
spc = pyspeckit.Cube(cube=cube)
header = cube.header
freq = (header['RESTFREQ']/1e9) * u.GHz
wcsspec = WCS(header).spectral
wcscel = WCS(header).celestial
np.size(velaxis)
chanlims = [wcsspec.world_to_pixel(velinit).tolist(), wcsspec.world_to_pixel(velend).tolist()]
rms = np.nanstd(np.vstack([spc.cube[:int(np.min(chanlims))], spc.cube[int(np.max(chanlims)):]]))
rmsmap = np.ones(np.shape(spc.cube)) * rms
spc.errorcube = rmsmap

momentsfile = cubefile+'_fitcube2g_moments.fits'
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


"""
1 gaussian
"""
mom01 = np.where(spc.momentcube[0]>rms*snratio, spc.momentcube[0],rms*snratio)
initguesses = np.array([mom01,spc.momentcube[1],spc.momentcube[2]])
fitfile1 = cubefile + '_1G_fitparams.fits'
if not os.path.exists(fitfile1):
    try:
        spc.fiteach(fittype='gaussian',
                    guesses=initguesses,
                    # negamp=False,
                    verbose=1,
                    signal_cut=signal_cut,
                    blank_value=np.nan,
                    start_from_point=(starting_point))
    # except ValueError:
    #     print('There are non-finite parameters in the fit')
    except AssertionError:
        print('There are non-finite parameters in the fit')
    spc.write_fit(fitfile1)
else:
    spc.load_model_fit(fitfile1, 3, fittype='gaussian')

'''
2 gaussians
'''
mom11 = spc.momentcube[1] - 1.
mom12 = spc.momentcube[1]
mom21 = mom22 = np.sqrt(spc.momentcube[2])/2

mom01 = mom02 = np.where(spc.momentcube[0]>rms*snratio, spc.momentcube[0],rms*snratio)
# mom11 = spc.momentcube[1] - 0.2
# mom12 = spc.momentcube[1] + 0.2
initguesses2 = np.concatenate([[mom01, mom11, mom21], [mom02, mom12, mom22]])

fitfile2 = cubefile + '_2G_fitparams.fits'
if os.path.exists(fitfile2):
    spc2.load_model_fit(fitfile2, 3, npeaks=2, fittype='gaussian')
else:
    try:
        spc2.fiteach(fittype='gaussian',
                    guesses=initguesses2,
                    negamp=False,
                    verbose=1,
                    signal_cut=signal_cut,
                    blank_value=np.nan,
                    start_from_point=(starting_point))
    except AssertionError:
        print('There are non-finite parameters in the fit')
    spc2.write_fit(fitfile2, overwrite=True)

spc = filter1G(spc,rms,snratio)
spc2 = filter2G(spc2,rms,snratio)

newheadaic = wcscel.to_header()
fitfile1filtered = cubefile + '_1G_fitparams_filtered.fits'
if not os.path.exists(fitfile1filtered):
    spc.write_fit(fitfile1filtered)
fitfile2filtered = cubefile + '_2G_fitparams_filtered.fits'
if not os.path.exists(fitfile2filtered):
    spc2.write_fit(fitfile2filtered)

fittedmodel1 = spc.get_modelcube()
fittedmodel2 = spc2.get_modelcube()

aic1map = np.zeros(np.shape(mom01)) *np.nan
aic2map = np.zeros(np.shape(mom01))*np.nan

totalpix = n_x*n_y
flag_prob = np.zeros(np.shape(mom01))

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
        if np.all(np.isnan(params_1G)) or np.all(np.isnan(params_2G)):
            # If one of the fits failed, the comparison does not make sense
            continue
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

# modelhdu = fits.PrimaryHDU(data=fittedmodel, header=header)
# modelhdu.writeto(cubefile + '_fitcube2g_fitted.fits', overwrite=True)
# spc.mapplot()
# spc.show_fit_param(1, cmap='inferno')
#
# spectra1 = spc.get_spectrum(34,27)
# spectra1.plotter()
# plt.show()
# spc.cube
#
# tmax, vlsr, sigmav = spc.parcube
# key_list = ['NAXIS3', 'CRPIX3', 'CDELT3', 'CUNIT3', 'CTYPE3', 'CRVAL3']
#
# commonhead = fits.getheader(fitfile)
# for key_i in key_list:
#     commonhead.remove(key_i)
# commonhead['NAXIS'] = 2
# commonhead['WCSAXES'] = 2
#
# hdutmax = fits.PrimaryHDU(data=tmax, header=commonhead)
# hdutmax.writeto(cubefile + '_2G_tmax.fits', overwrite=True)
# headervelocities = commonhead.copy()
# headervelocities['BUNIT'] = 'km/s'
# hduvlsr = fits.PrimaryHDU(data=vlsr, header=headervelocities)
# hduvlsr.writeto(cubefile + '_2G_Vc.fits', overwrite=True)
# hdusigmav = fits.PrimaryHDU(data=sigmav, header=headervelocities)
# hdusigmav.writeto(cubefile + '_2G_sigma_v.fits', overwrite=True)
#
#
# plt.show()
