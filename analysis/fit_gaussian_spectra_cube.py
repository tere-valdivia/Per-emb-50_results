"""
Author: Teresa Valdivia-Mena
Last revised August 30, 2022

This code is to fit one Gaussian to each spectra above a certain signal to noise
level for H2CO (303-202) and C18O (2-1) emission cubes.

Current state: C18O
"""

from astropy.coordinates import SkyCoord
from astropy.modeling.functional_models import Gaussian2D
import os
import regions
from astropy.wcs import WCS
from astropy.io import fits
import numpy as np
import matplotlib.pylab as plt
import pyspeckit
from spectral_cube import SpectralCube

import sys
sys.path.append('../')
from NOEMAsetup import *


# Files to use
# cubefile = '../' + H2CO_303_202_s
cubefile = '../' + C18O_2_1_s # '../C18O/CDconfig/JEP/JEP_mask_multi_Per-emb-50_CD_l025l064_uvsub_C18O_pbcor'

# Define the velocities where there is emission to calculate the rms
# For pbcor, we need to give it the rms
# Where we estimate the line is
velinit = 5.5 * u.km/u.s
velend = 9.5 * u.km/u.s

# Region where we want to fit: it is a square that is smaller than the "small"
# cube

fitregionfile = 'C18O_fitregion.reg' # 'H2CO_fitregion.reg'

# starting_point = (70, 82)  # H2CO
# starting_point = (53,116)
starting_point = (133,104) #C18O

if not os.path.exists(cubefile+'_fitcube.fits'):
    # The cube must be in K and km/s
    cube1 = SpectralCube.read(cubefile+'.fits').with_spectral_unit(u.km/u.s)
    header1 = cube1.header
    restfreq = header1['restfreq'] * u.Hz
    beamarea = 1.133 * header1['bmaj'] * u.deg * header1['bmin'] * u.deg
    cube1 = cube1.to(u.K)
    regionlist = regions.read_ds9(fitregionfile)
    subcube = cube1.subcube_from_regions(regionlist)
    subcube.hdu.writeto(cubefile+'_fitcube.fits')

spc = pyspeckit.Cube(cubefile+'_fitcube.fits')
header = spc.header
ra = header['ra']  # phasecent
dec = header['dec']
naxis = header['naxis1']
freq = (header['RESTFREQ']/1e9) * u.GHz
wcsspec = WCS(header).spectral
wcscel = WCS(header).celestial
chanlims = [wcsspec.world_to_pixel(velinit).tolist(), wcsspec.world_to_pixel(velend).tolist()]

# only for the rms
rms = np.nanstd(np.vstack([spc.cube[:int(np.min(chanlims))], spc.cube[int(np.max(chanlims)):]]))
rmsmap = np.ones(np.shape(spc.cube)) * rms
print(rms, spc.unit)

momentsfile = cubefile+'_fitcube_moments.fits'
if os.path.exists(momentsfile):
    spc.momentcube = fits.getdata(momentsfile)
else:
    spc.momenteach(vheight=False)
    moments = fits.PrimaryHDU(data=spc.momentcube, header=header)
    moments.writeto(momentsfile)


def filter(spc, rms, rmslevel, velinit, velend, negative=True, errorfrac=0.5, epsilon=1.e-5):
    """
    Replace the pixels in the fitted cube with np.nan where the fit is not
    good enough according to our criteria.

    The criteria that a pixel must have are:
    - The error is not zero
    - The value must not be negative (in this case we know the moment 1 must be
    positive, so we specify negative=True, can be changed)
    - The error fraction is lower than errorfrac
    - The moment 1 value must be within the range [velinit,velend]
    - The peak value must be larger than rms times rmslevel
    - The weighted velocity dispersion must be smaller than the absolute
    value of velend-velinit
    - If one pixel in a spectra is np.nan, all the spectra must be nan (sanity
    check)

    Args:
        spc (pyspeckit.Cube): pyspeckit cube with the data already fitted
        rms (float): rms noise level of the cube in K
        rmslevel (float): signal-to-noise ratio for the desired threshold
        for the peak values
        velinit, velend (float): initial and final velocity where the central
        position of the lines are estimated to be
        negative (bool): if True, filter out fits with negative peaks. Default
        is True
        errorfrac (float): maximum fraction between the uncertainty of a
        parameter and the value of that parameter. All spectra with any error
        fraction larger than this value will be filtered
        epsilon (float): tolerance for the error. If the uncertainty is below
        this value, the spectrum will be filtered


    Returns:
        pyspeckit.Cube: pyspeckit cube with its parcube and errcube modified
        to fit the criteria

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

    velocitymask = np.where(spc.parcube[1] < velinit.value, 1, 0) + \
        np.where(spc.parcube[1] > velend.value, 1, 0)
    spc.parcube[np.where(np.repeat([velocitymask], 3, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([velocitymask], 3, axis=0))] = np.nan

    peakmask = np.where(spc.parcube[0] < rmslevel*rms, 1, 0)
    spc.parcube[np.where(np.repeat([peakmask], 3, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([peakmask], 3, axis=0))] = np.nan

    sigmamask = np.where(spc.parcube[2] > (velend-velinit).value/2, 1, 0) + \
        np.where(spc.parcube[2] < np.abs(header['cdelt3']), 1, 0)
    spc.parcube[np.where(np.repeat([sigmamask], 3, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([sigmamask], 3, axis=0))] = np.nan

    # Force if one pixel in a channel is nan, all the same pixels
    # in all channels must be nan
    nanmask = np.sum(np.where(np.isnan(np.concatenate([spc.parcube, spc.errcube])), 1, 0), axis=0)
    spc.parcube[np.where(np.repeat([nanmask], 3, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([nanmask], 3, axis=0))] = np.nan

    return spc


fitfile = cubefile + '_fitcube_1G_fitparams.fits'
if os.path.exists(fitfile):
    spc.load_model_fit(fitfile, 3, fittype='gaussian')
    fittedmodel = spc.get_modelcube()
else:
    # running time: about 5 minutes for H2CO
    spc.fiteach(fittype='gaussian',
                use_neighbor_as_guess=True,
                guesses=spc.momentcube,
                verbose_level=0,
                verbose=0,
                errmap=rmsmap,
                signal_cut=4,
                prevalidate_guesses=True,
                blank_value=np.nan,
                start_from_point=(starting_point))
    spc.write_fit(fitfile)
    fittedmodel = spc.get_modelcube()

fitfile_filtered = cubefile + '_fitcube_1G_fitparams_filtered.fits'
if os.path.exists(fitfile_filtered):
    spc.load_model_fit(fitfile_filtered, 3, fittype='gaussian')
else:
    spc = filter(spc, rms, 4, velinit, velend)
    spc.write_fit(fitfile_filtered)

tmax, vlsr, sigmav = spc.parcube
key_list = ['NAXIS3', 'CRPIX3', 'CDELT3', 'CUNIT3', 'CTYPE3', 'CRVAL3']

commonhead = fits.getheader(fitfile)
for key_i in key_list:
    commonhead.remove(key_i)
commonhead['NAXIS'] = 2
commonhead['WCSAXES'] = 2


if not os.path.exists(cubefile + '_fitcube_1G_tmax.fits'):
    hdutmax = fits.PrimaryHDU(data=tmax, header=commonhead)
    hdutmax.writeto(cubefile + '_fitcube_1G_tmax.fits', overwrite=True)

headervelocities = commonhead.copy()
headervelocities['BUNIT'] = 'km/s'

if not os.path.exists(cubefile + '_fitcube_1G_Vc.fits'):
    hduvlsr = fits.PrimaryHDU(data=vlsr, header=headervelocities)
    hduvlsr.writeto(cubefile + '_fitcube_1G_Vc.fits', overwrite=True)
if not os.path.exists(cubefile + '_fitcube_1G_sigma_v.fits'):
    hdusigmav = fits.PrimaryHDU(data=sigmav, header=headervelocities)
    hdusigmav.writeto(cubefile + '_fitcube_1G_sigma_v.fits', overwrite=True)
