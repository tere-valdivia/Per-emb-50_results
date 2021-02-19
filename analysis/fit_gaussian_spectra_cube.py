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

# Define the velocities where there is emission to calculate the rms

cubefile = H2CO_303_202_s
# Where we estimate the line is
velinit = 5.5 * u.km/u.s
velend = 9.5 * u.km/u.s
fitregionfile = '../analysis/H2CO_fitregion.reg'
starting_point = (70, 82)

if not os.path.exists('../'+cubefile+'_fitcube.fits'):
    # The cube to fit must be smaller than the small cube we set earlier
    # and must be in K and km/s
    cube1 = SpectralCube.read(cubefile+'.fits').with_spectral_unit(u.km/u.s)
    header1 = cube1.header
    restfreq = header1['restfreq'] * u.Hz
    beamarea = 1.133 * header1['bmaj'] * u.deg * header1['bmin'] * u.deg
    cube1 = cube1.to(u.K, u.brightness_temperature(restfreq, beam_area=beamarea))
    regionlist = regions.read_ds9(fitregionfile)
    subcube = cube1.subcube_from_regions(regionlist)
    subcube.hdu.writeto(cubefile+'_fitcube.fits')

spc = pyspeckit.Cube('../'+cubefile+'_fitcube.fits')
header = spc.header
ra = header['ra']
dec = header['dec']
naxis = header['naxis1']
wcsspec = WCS(header).spectral
wcscel = WCS(header).celestial
chanlims = [wcsspec.world_to_pixel(velinit).tolist(), wcsspec.world_to_pixel(velend).tolist()]
rms = np.std(np.vstack([spc.cube[:int(np.min(chanlims))], spc.cube[int(np.max(chanlims)):]]))
rmsmap = np.ones(np.shape(spc.cube)) * rms

momentsfile = cubefile+'_fitcube_moments.fits'

if os.path.exists(momentsfile):
    spc.momentcube = fits.getdata(momentsfile)
else:
    spc.momenteach(vheight=False)
    moments = fits.PrimaryHDU(data=spc.momentcube, header=header)
    moments.writeto(momentsfile)


def filter(spc, rms, rmslevel, velinit, velend, negative=True, errorfrac=0.5, epsilon=1.e-5):
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
    # in all  channels must be nan
    nanmask = np.sum(np.where(np.isnan(np.concatenate([spc.parcube, spc.errcube])), 1, 0), axis=0)
    spc.parcube[np.where(np.repeat([nanmask], 3, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([nanmask], 3, axis=0))] = np.nan
    return spc

fitfile = cubefile + '_1G_fitparams.fits'
if os.path.exists(fitfile):
    spc.load_model_fit(fitfile, 3, fittype='gaussian')
    spc = filter(spc, rms, 4, velinit, velend)
    fittedmodel = spc.get_modelcube()
    spc.write_fit(fitfile, overwrite=True)
else:
    spc.fiteach(fittype='gaussian',
                use_neighbor_as_guess=True,
                guesses=spc.momentcube,
                verbose=1,
                errmap=rmsmap,
                signal_cut=4,
                blank_value=np.nan,
                start_from_point=(starting_point))
    spc = filter(spc, rms, 4, velinit, velend)
    spc.write_fit(fitfile)
    fittedmodel = spc.get_modelcube()

# # TODO: Separate the cube into 2D fits each with one parameter
tmax, vlsr, sigmav = spc.parcube
key_list = ['NAXIS3', 'CRPIX3', 'CDELT3', 'CUNIT3', 'CTYPE3', 'CRVAL3']

commonhead = fits.getheader(fitfile)
for key_i in key_list:
    commonhead.remove(key_i)
commonhead['NAXIS'] = 2
commonhead['WCSAXES'] = 2

hdutmax = fits.PrimaryHDU(data=tmax, header=commonhead)
hdutmax.writeto(cubefile + '_1G_tmax.fits', overwrite=True)
headervelocities = commonhead.copy()
headervelocities['BUNIT'] = 'km/s'
hduvlsr = fits.PrimaryHDU(data=vlsr, header=headervelocities)
hduvlsr.writeto(cubefile + '_1G_Vc.fits', overwrite=True)
hdusigmav = fits.PrimaryHDU(data=sigmav, header=headervelocities)
hdusigmav.writeto(cubefile + '_1G_sigma_v.fits', overwrite=True)

modelhdu = fits.PrimaryHDU(data=fittedmodel, header=header)
modelhdu.writeto(cubefile + '_fitcube_fitted.fits', overwrite=True)
spc.mapplot()
spc.show_fit_param(2, cmap='inferno')

plt.show()
