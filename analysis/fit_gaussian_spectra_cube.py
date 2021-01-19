from NOEMAsetup import *
from spectral_cube import SpectralCube
import pyspeckit
import matplotlib.pylab as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import regions
import os
# Define the velocities where there is emission

cubefile = H2CO_303_202_s
# Where we estimate the line is
velinit = 5500 #*u.m/u.s
velend = 9500 #*u.m/u.s
fitregionfile = 'analysis/H2CO_fitregion.reg'
starting_point = (70,82)

if not os.path.exists(cubefile+'_fitcube.fits'):
    cube = SpectralCube.read(cubefile+'.fits')
    regionlist = regions.read_ds9(fitregionfile)
    subcube = cube.subcube_from_regions(regionlist)
    subcube.hdu.writeto(cubefile+'_fitcube.fits')

spc = pyspeckit.Cube(cubefile+'_fitcube.fits')
header = spc.header
ra = header['ra']
dec = header['dec']
naxis = header['naxis1']
wcsspec = WCS(header).spectral
chanlims = wcsspec.all_world2pix([velinit, velend], 0)[0]
rmsmap = np.sqrt(np.mean(((np.vstack([spc.cube[:int(np.min(chanlims))], spc.cube[int(np.max(chanlims)):]]))**2), axis=0))
plt.imshow(rmsmap)

if os.path.exists(cubefile+'_fitcube_moments.fits'):
    spc.momentcube = fits.getdata(cubefile+'_fitcube_moments.fits')
else:
    spc.momenteach(vheight=False)
    momentsfile = fits.PrimaryHDU(data=spc.momentcube, header=header)
    momentsfile.writeto(cubefile+'_fitcube_moments.fits')

spc.fiteach(fittype = 'gaussian',
            guesses = spc.momentcube,
            errmap = rmsmap,
            signal_cut = 4,
            start_from_point=starting_point) # ignore pixels with SNR<4)
# spc.write_fit(cubefile + '_1G_fitparams.fits')
spc.mapplot()
plt.imshow(spc.errcube[0])
# show the fitted amplitude
spc.show_fit_param(0, cmap='viridis')
plt.show()
