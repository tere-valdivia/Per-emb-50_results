import numpy as np
from spectral_cube import SpectralCube
import astropy.units as u
from NOEMAsetup import *
from astropy.wcs import WCS

cubefile = SO_55_44
velcut = 0
# If you want to cut in velocity as well
if velcut:
    velmin = -4000
    velmax = 16000

cube = SpectralCube.read(cubefile+'.fits')
# cube = SpectralCube.read(cubefile+'_pbcor.fits')
header = cube.header
wcsspec = WCS(header).spectral
vellims = wcsspec.all_world2pix([velmin, velmax], 0)
pixsize = header['CDELT2'] * u.deg
naxis = header['NAXIS1']
midaxis = int(naxis/2)

# Through radius selection
radius_ex = 20 * u.arcsec
# assuming the celestial plane is a square
radius_pix = int((radius_ex.to(u.deg)/pixsize).value)
if velcut:
    sub_cube = cube[int(np.min(vellims)):int(np.max(vellims)), midaxis-radius_pix:midaxis+radius_pix, midaxis-radius_pix:midaxis+radius_pix]
else:
    sub_cube = cube[:, midaxis-radius_pix:midaxis+radius_pix, midaxis-radius_pix:midaxis+radius_pix]
sub_cube.hdu.writeto(cubefile+'_small.fits')


# Through minimal masking
# mask = np.abs(cube) > 1e-9
# cube = cube.with_mask(mask)
# sub_cube = cube[int(np.min(vellims)):int(np.max(vellims)), :, :].minimal_subcube()
# sub_cube.hdu.writeto(cubefile+'_pbcor_small.fits')
