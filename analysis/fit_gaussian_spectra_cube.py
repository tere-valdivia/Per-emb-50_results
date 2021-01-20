from NOEMAsetup import *
from spectral_cube import SpectralCube
import pyspeckit
import matplotlib.pylab as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import regions
import os

'''
This code does not fit with uncertainties in the parameters.
I am trying to fix this right now

Edit: it is throwing the Warning gnorm=0 with much less frequency now and not
immediately
'''
# Define the velocities where there is emission to calculate the rms

cubefile = H2CO_303_202_s
# Where we estimate the line is
velinit = 5.5  *u.km/u.s
velend = 9.5  *u.km/u.s
fitregionfile = 'analysis/H2CO_fitregion.reg'
starting_point = (70,82)

if not os.path.exists(cubefile+'_fitcube.fits'):
    # The cube to fit must be smaller than the small cube we set earlier
    # and must be in K and km/s
    cube1 = SpectralCube.read(cubefile+'.fits').with_spectral_unit(u.km/u.s)
    header1 = cube1.header
    restfreq = header1['restfreq'] * u.Hz
    beamarea = 1.133 * header1['bmaj'] * u.deg *header1['bmin'] * u.deg
    cube1 = cube1.to(u.K, u.brightness_temperature(restfreq, beam_area=beamarea))
    regionlist = regions.read_ds9(fitregionfile)
    subcube = cube1.subcube_from_regions(regionlist)
    subcube.hdu.writeto(cubefile+'_fitcube.fits')

spc = pyspeckit.Cube(cubefile+'_fitcube.fits')
header = spc.header
ra = header['ra']
dec = header['dec']
naxis = header['naxis1']
wcsspec = WCS(header).spectral
# chanlims = wcsspec.all_world2pix([velinit, velend], 0)[0]
chanlims = [wcsspec.world_to_pixel(velinit).tolist(), wcsspec.world_to_pixel(velend).tolist()]

# rmsmap = np.sqrt(np.mean(((np.vstack([spc.cube[:int(np.min(chanlims))], spc.cube[int(np.max(chanlims)):]]))**2), axis=0))
rmsmap = np.ones(np.shape(spc.cube))*np.std(np.vstack([spc.cube[:int(np.min(chanlims))], spc.cube[int(np.max(chanlims)):]]))

momentsfile = cubefile+'_fitcube_moments.fits'

if os.path.exists(momentsfile):
    spc.momentcube = fits.getdata(momentsfile)
else:
    spc.momenteach(vheight=False)
    moments = fits.PrimaryHDU(data=spc.momentcube, header=header)
    moments.writeto(momentsfile)

fitfile = cubefile + '_1G_fitparams.fits'
if os.path.exists(fitfile):
    spc.load_model_fit(fitfile, 3, fittype='gaussian')
else:
    spc.fiteach(fittype = 'gaussian',
                use_neighbor_as_guess=True,
                guesses = spc.momentcube,
                verbose = 1,
                errmap = rmsmap,
                signal_cut= 4,
                blank_value=np.nan,
                start_from_point=(starting_point)) # ignore pixels with SNR<4)
    spc.write_fit(fitfile)

spc.mapplot()
# show the fitted amplitude
spc.show_fit_param(0, cmap='viridis')

plt.show()
