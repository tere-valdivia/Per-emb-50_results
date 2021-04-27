import numpy as np
from spectral_cube import SpectralCube
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import sys
sys.path.append('../')
from ChauvenetRMS import *
from NOEMAsetup import *
import os
import regions
import matplotlib.pyplot as plt

filenameH2CO = '../' + H2CO_303_202_s
filenameC18O = '../' + C18O_2_1
snratio = 3
rms = 13.94 * u.mJy/u.beam

# reproject the C18O to the H2CO wcs
# do it with the small so you can mask the small cube
# TODO: Change the unit to K
cubeC18O = SpectralCube.read(filenameC18O+'.fits').with_spectral_unit(u.km/u.s)
cubeH2CO = SpectralCube.read(filenameH2CO+'.fits').with_spectral_unit(u.km/u.s).spectral_slab(5.5*u.km/u.s,9.5*u.km/u.s)
spectral_grid_objective = cubeH2CO.spectral_axis
cubeC18O = cubeC18O.spectral_interpolate(spectral_grid_objective)
cubeC18O = cubeC18O.reproject(cubeH2CO.header) #beam is still there
if not os.path.exists(filenameC18O+'_reprojectH2COs.fits'):
    cubeC18O.write(filenameC18O+'_reprojectH2COs.fits')
# mask the cube where there is emission
masked_cube = cubeC18O.with_mask(cubeH2CO > snratio* rms)

# Now leave out all that is not streamer
region_streamer = '../data/region_streamer_l.reg'
regio = regions.read_ds9(region_streamer)
streamer_cube = masked_cube.subcube_from_regions(regio)
if not os.path.exists(filenameC18O+'_H2COmasked_'+str(snratio)+'sigma.fits'):
    streamer_cube.write(filenameC18O+'_H2COmasked_'+str(snratio)+'sigma.fits')
newheaderC18O = cubeC18O.header

# Change cube to K
k_streamer_cube = streamer_cube.to(u.K)

# do a moment 1
mom0 = k_streamer_cube.moment(order=0)
wcsmom = WCS(newheaderC18O).celestial


# Do a N(C18O) map
# for now,lets assume a constant Tex
Tex = 10 * u.K
B0 = 54891.420 * u.MHz
NC18O = N_C18O_21(mom0, B0, Tex)

# TODO: calculate Tex for each pixel (12CO?)
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection=wcsmom)
im = ax.imshow(NC18O.value)
fig.colorbar(im,ax=ax)
