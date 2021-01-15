import numpy as np
from spectral_cube import SpectralCube
import astropy.units as u
from NOEMAsetup import *
from astropy.wcs import WCS

cubefile = C18O_2_1

# Through radius selection
# radius_ex = 20 * u.arcsec
# cube = SpectralCube.read(cubefile+'.fits')
# header = cube.header
# pixsize = header['CDELT2'] * u.deg
# naxis = header['NAXIS1']
# midaxis = int(naxis/2)
# # assuming the celestial plane is a square
# radius_pix = int((radius_ex.to(u.deg)/pixsize).value)
# sub_cube = cube[:, midaxis-radius_pix:midaxis+radius_pix, midaxis-radius_pix:midaxis+radius_pix]
# sub_cube.hdu.writeto(cubefile+'_small.fits')
#

# Through minimal masking
cube = SpectralCube.read(cubefile+'_pbcor.fits')
mask = np.abs(cube) > 1e-9
cube = cube.with_mask(mask)
sub_cube = cube.minimal_subcube()
sub_cube.hdu.writeto(cubefile+'_pbcor_small.fits')
