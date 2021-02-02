from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from spectral_cube import SpectralCube
from NOEMAsetup import *
import os

'''
This code computes the moments 0, 1 and 8 and the linewidth map for the cube which
is used to fit gaussian profiles later and of the complete cubes
'''
mask = False
velinit = 5.5 * u.km/u.s
# velend = 8.0 * u.km/u.s  # streamline feature
velend = 9.0 * u.km/u.s # total emission
# rangename = 'stream'
rangename = 'total'

# filename = H2CO_303_202_s + '_fitcube'
filename = H2CO_303_202

if os.path.exists(filename+'.fits'):
    cube = SpectralCube.read(filename+'.fits').with_spectral_unit(u.km/u.s)
    header = cube.header
    wcsspec = WCS(header).spectral
    chanlims = [int(wcsspec.world_to_pixel(velinit).tolist()),
                int(wcsspec.world_to_pixel(velend).tolist())]
    rms = np.sqrt(np.mean((np.vstack(
        [cube.unmasked_data[:np.min(chanlims), :, :], cube.unmasked_data[np.max(chanlims):, :, :]]))**2))
    subcube = cube.spectral_slab(velinit, velend)
    if mask:
        subcube = subcube.with_mask((subcube > 4 * rms))

    for i in [0, 1, 2, 8]:
        if i == 2:
            moment = subcube.linewidth_sigma()
        elif i == 8:
            moment = subcube.max(axis=0)
        else:
            moment = subcube.moment(order=i)
        savename = filename+'_'+rangename+'_sigma.fits' if i == 2 else filename + \
            '_'+rangename+'_mom'+str(i)+'.fits'
        try:
            moment.write(savename)
        except OSError:
            confirm = input('The file ' + savename +
                            ' already exists. Do you want to overwrite it? [y/n]')
            if confirm == 'y':
                moment.write(savename, overwrite=True)
            else:
                continue
else:
    print('The file '+filename+' does not exist. You have to generate the _fitcube first, which is generated from the _small cube when you fit a gaussian profile in the code fit_gaussian_spectra_cube.py.')
