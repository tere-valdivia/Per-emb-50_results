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
This code is to fit two gaussian components in the cube.
"""

pbcor = False
cubefile = '../SO_55_44/CDconfigsmall/Per-emb-50_CD_l009l048_uvsub_SO_multi_small'
# cubefile_nonpb = '../' + H2CO_303_202_s
# cubefile = '../C18O/CDconfig/JEP/JEP_mask_multi_Per-emb-50_CD_l025l064_uvsub_C18O_pbcor'
# cubefile_nonpb = '../C18O/CDconfig/JEP/JEP_mask_multi_Per-emb-50_CD_l025l064_uvsub_C18O'
# Where we estimate the line is
velinit = 5.5 * u.km/u.s
velend = 9.5 * u.km/u.s
# fitregionfile = '../analysis/H2CO_fitregion.reg'
fitregionfile = 'SO_2G_fitregion.reg'
# starting_point = (70, 82) #H2CO
starting_point = (34, 27)
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


spc = pyspeckit.Cube(cubefile+'_fitcube2g.fits')
header = spc.header
ra = header['ra']  # phasecent
dec = header['dec']
naxis = header['naxis1']
freq = (header['RESTFREQ']/1e9) * u.GHz
wcsspec = WCS(header).spectral
wcscel = WCS(header).celestial
chanlims = [wcsspec.world_to_pixel(velinit).tolist(), wcsspec.world_to_pixel(velend).tolist()]

# TODO: Implement for pbcorrected images
rms = np.nanstd(np.vstack([spc.cube[:int(np.min(chanlims))], spc.cube[int(np.max(chanlims)):]]))
rmsmap = np.ones(np.shape(spc.cube)) * rms

momentsfile = cubefile+'_fitcube2g_moments.fits'
if os.path.exists(momentsfile):
    spc.momentcube = fits.getdata(momentsfile)
else:
    spc.momenteach(vheight=False)
    moments = fits.PrimaryHDU(data=spc.momentcube, header=header)
    moments.writeto(momentsfile)

# We use the moments to do initial guesses
# gaussian component 1: strongest
# as we want to fit two gaussians, we need a fittype='gaussian' with 6
# parameters
# moment1 = spc.momentcube[1]
# moment2 = spc.momentcube[1]
# gauss1vc0 = moment1-moment2
# gauss2vc0 = moment1+moment2
# gauss2peak0 = spc.momentcube[0]/2
# initguesses = np.concatenate([[spc.momentcube[0], gauss1vc0, moment2],
#                               [gauss2peak0, gauss2vc0, moment2]])
initguesses = np.concatenate([spc.momentcube, spc.momentcube])


def filter(spc, rms, rmslevel, velinit, velend, negative=True, errorfrac=0.5, epsilon=1.e-5):
    """
    Replace the pixels in the fitted cube with np.nan where the fit is not
    good enough according to our criteria.

    The criteria that a pixel must have are:
    - The error is not zero
    - The value for each peak must not be negative (in this case we know the
    moment 1 must be positive, so we specify negative=True, can be changed)
    - The error fraction is lower than errorfrac
    - The moment 1 value must be within the range [velinit,velend]
    - The peak value for both gaussian fits must be larger than rms times rmslevel
    - The weighted velocity dispersion must be smaller than the absolute
    value of velend-velinit
    - If one pixel in a spectra is np.nan, all the spectra must be nan (sanity
    check)

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
    # maximum acceptable error fraction
    errormask = np.where(np.abs(spc.errcube[0]/spc.parcube[0]) > errorfrac, 1, 0)\
        + np.where(np.abs(spc.errcube[1]/spc.parcube[1]) > errorfrac, 1, 0)\
        + np.where(np.abs(spc.errcube[2]/spc.parcube[2]) > errorfrac, 1, 0)\
        + np.where(np.abs(spc.errcube[3]/spc.parcube[3]) > errorfrac, 1, 0)\
        + np.where(np.abs(spc.errcube[4]/spc.parcube[4]) > errorfrac, 1, 0)\
        + np.where(np.abs(spc.errcube[5]/spc.parcube[5]) > errorfrac, 1, 0)
    spc.parcube[np.where(np.repeat([errormask], 6, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([errormask], 6, axis=0))] = np.nan
    velocitymask = np.where(spc.parcube[1] < velinit.value, 1, 0) + \
        np.where(spc.parcube[1] > velend.value, 1, 0) + \
        np.where(spc.parcube[4] > velend.value, 1, 0) + \
        np.where(spc.parcube[4] < velinit.value, 1, 0)
    spc.parcube[np.where(np.repeat([velocitymask], 6, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([velocitymask], 6, axis=0))] = np.nan
    peakmask = np.where(spc.parcube[0] > np.amax(spc.cube), 1, 0) + \
                np.where(spc.parcube[3] > np.amax(spc.cube), 1, 0)
    # peakmask = np.where(spc.parcube[0] < rmslevel*rms, 1, 0) + \
    #     np.where(spc.parcube[3] < rmslevel*rms, 1, 0)
    spc.parcube[np.where(np.repeat([peakmask], 6, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([peakmask], 6, axis=0))] = np.nan
    # sigmamask = np.where(spc.parcube[2] > (velend-velinit).value/2, 1, 0) + \
    #     np.where(spc.parcube[2] < np.abs(header['cdelt3']), 1, 0) + \
    #     np.where(spc.parcube[5] > (velend-velinit).value/2, 1, 0) + \
    #     np.where(spc.parcube[5] < np.abs(header['cdelt3']), 1, 0)
    # spc.parcube[np.where(np.repeat([sigmamask], 6, axis=0))] = np.nan
    # spc.errcube[np.where(np.repeat([sigmamask], 6, axis=0))] = np.nan
    # Force if one pixel in a channel is nan, all the same pixels
    # in all  channels must be nan
    # nanmask = np.sum(np.where(np.isnan(np.concatenate([spc.parcube, spc.errcube])), 1, 0), axis=0)
    # spc.parcube[np.where(np.repeat([nanmask], 6, axis=0))] = np.nan
    # spc.errcube[np.where(np.repeat([nanmask], 6, axis=0))] = np.nan
    return spc


fitfile = cubefile + '_2G_fitparams.fits'
if os.path.exists(fitfile):
    spc.load_model_fit(fitfile, 3, npeaks=2,fittype='gaussian')
    spc = filter(spc, rms, 3, velinit, velend)
    fittedmodel = spc.get_modelcube()
    spc.write_fit(fitfile, overwrite=True)
else:
    spc.fiteach(fittype='gaussian',
                use_neighbor_as_guess=True,
                guesses=initguesses,
                verbose=1,
                errmap=rmsmap,
                signal_cut=4,
                blank_value=np.nan,
                start_from_point=(starting_point))
    spc = filter(spc, rms, 3, velinit, velend)
    spc.write_fit(fitfile, overwrite=True)
    fittedmodel = spc.get_modelcube()

modelhdu = fits.PrimaryHDU(data=fittedmodel, header=header)
modelhdu.writeto(cubefile + '_fitcube2g_fitted.fits', overwrite=True)
spc.mapplot()
spc.show_fit_param(1, cmap='inferno')

spectra1 = spc.get_spectrum(34,27)
spectra1.plotter()
plt.show()
spc.cube
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
