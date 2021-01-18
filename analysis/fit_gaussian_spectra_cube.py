from NOEMAsetup import *
from spectral_cube import SpectralCube
import pyspeckit
import matplotlib.pylab as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import regions
# Define the velocities where there is emission

cubefile = H2CO_303_202_s
# Where we estimate the line is
velinit = 5.5 *u.km/u.s
velend = 8.5 *u.km/u.s
fitregionfile = 'analysis/H2CO_fitregion.reg'

cube = SpectralCube.read(cubefile+'.fits')
regionlist = regions.read_ds9(fitregionfile)
subcube = cube.subcube_from_regions(regionlist)

spc = pyspeckit.Cube(cube=subcube)
header = spc.header
ra = header['ra']
dec = header['dec']
naxis = header['naxis1']
wcsspec = WCS(header).spectral
chanlims = wcsspec.all_world2pix([velinit.to(u.m/u.s).value, velend.to(u.m/u.s).value], 0, ra_dec_order=False)[0]
rms_map = np.sqrt(np.mean((np.vstack([spc.cube[:int(np.min(chanlims))], spc.cube[int(np.max(chanlims)):]])**2), axis=0))
plt.imshow(rms_map)
# get a cube of moments
spc.momenteach(vheight=False)
spc.fiteach(fittype = 'gaussian',
            guesses = spc.momentcube,
            errmap = rms_map,
            signal_cut = 3, # ignore pixels with SNR<3
            blank_value = np.nan)
spc.write_fit(cubefile + '_1G_fitparams.fits')
spc.mapplot()
# show the fitted amplitude
spc.show_fit_param(0, cmap='viridis')

plt.show()
