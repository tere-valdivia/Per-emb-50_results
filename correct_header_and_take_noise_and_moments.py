from ChauvenetRMS import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.font_manager as fm
from spectral_cube import SpectralCube

"""
The imaged data from mapping do not have the units in the header.
This code corrects the header and does a quick calculation of the rms and the
moments 0, 1 and 2 of the image

"""

folder = 'C18O/CDconfig/JEP/'
cubename = folder + 'JEP_mask_multi_Per-emb-50_CD_l025l064_uvsub_C18O'
emptyheader = False

velstart = -1  # km/s
velend = 14  # km/s


logfile = open(cubename+'_loginfo.txt', 'w')
logfile.write('Basic information:\n')

# correct header

if emptyheader:
    fits.setval(cubename+'.fits', 'CUNIT1', value='deg     ')
    fits.setval(cubename+'.fits', 'CUNIT2', value='deg     ')
    fits.setval(cubename+'.fits', 'CUNIT3', value='m/s     ')

# Load correct cube

cube = SpectralCube.read(cubename+'.fits')
cube = cube.with_spectral_unit(u.km/u.s)
header = cube.hdu.header
restfreq = header['restfreq'] * u.Hz
bmaj = (header['bmaj'] * u.deg).to(u.arcsec)
bmin = (header['bmin'] * u.deg).to(u.arcsec)
logfile.write('Beam major = '+str(round(bmaj.value, 2))+' arcsec\n')
logfile.write('Beam minor = '+str(round(bmin.value, 2))+' arcsec\n')
logfile.write('Beam position angle = '+str(round(header['BPA'], 2))+' deg\n')
logfile.write('Delta_v = '+str(round(header['CDELT3'], 4))+' km/s \n')

logfile.write('\n')
logfile.write('---------------------------')
logfile.write('\n')

rms, __ = calculatenoise(cube.hdu.data)
beamarea = 1.133 * bmaj * bmin
rms_K = (rms * u.Jy/u.beam).to(u.K, u.brightness_temperature(restfreq, beam_area=beamarea))
logfile.write('The noise of the cube is '+str(round(rms*1000, 2))+' m('+header['BUNIT']+')\n')
logfile.write('or '+str(round(rms_K.value*1000, 2))+' mK\n')

subcube = cube.spectral_slab(velstart*u.km/u.s, velend*u.km/u.s)
rmssub, _ = calculatenoise(subcube.hdu.data)
logfile.write('The noise of the cube between '+str(velstart)+' and '+str(velend) +
              ' km/s is '+str(round(rmssub*1000, 2))+' m('+header['BUNIT']+')\n')
logfile.write('The peak of the cube is ' +
              str(round(cube.max().value*1000, 2))+' m('+header['BUNIT']+')\n')
logfile.write('\n')
logfile.write('---------------------------')
logfile.write('\n')

# Make moments

integrated = subcube.moment(order=0)
rmsmom0, _ = calculatenoise(integrated.value)
logfile.write('The peak of the integrated image between '+str(velstart)+' and ' +
              str(velend)+' km/s is '+str(round(integrated.max().value*1000, 2))+' m('+header['BUNIT']+') km/s'+'\n')
logfile.write('The noise of the integrated image between '+str(velstart)+' and ' +
              str(velend)+' km/s is '+str(round(rmsmom0*1000, 2))+' m('+header['BUNIT']+') km/s'+'\n')
integrated.write(cubename+'_integrated_python.fits', overwrite=True)

mask_vel = (subcube > 3*rms*u.Jy/u.beam)
subcube2 = subcube.with_mask(mask_vel)
velfield = subcube2.moment(order=1)
velfield.write(cubename+'_velocity_python.fits', overwrite=True)

velwidth = subcube2.moment(order=2)
velwidth.write(cubename+'_velwidth_python.fits', overwrite=True)
#
logfile.close()
