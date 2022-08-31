"""
Author: Teresa Valdivia-Mena
Last revised August 31, 2022

This code is a quick estimation of the central velocity of the protostar,
based on our C18O observations
"""

import sys
sys.path.append('../')

import matplotlib.pyplot as plt
from NOEMAsetup import *
from radio_beam import Beam
from astropy import units as u
import pyspeckit
from spectral_cube import SpectralCube
import numpy as np
from ChauvenetRMS import *

filename = '../' + C18O_2_1_s + '.fits'
cube = SpectralCube.read(filename).with_spectral_unit(u.km/u.s)
wcssky = cube.wcs.celestial
rms, _ = calculatenoise(cube.hdu.data)
kernel = cube.beam.as_kernel(cube.wcs.pixel_scale_matrix[1, 1]*u.deg)
x, y = wcssky.all_world2pix(ra_Per50, dec_Per50, 0)
kernsize = kernel.shape[0]
subcube = cube[:, int(y-kernsize/2.):int(y+kernsize/2.), int(x-kernsize/2.):int(x+kernsize/2.)]
mask = kernel.array > (0.01*kernel.array.max())

msubcube = subcube.with_mask(mask)
weighted_cube = msubcube * kernel.array
beam_weighted_spectrum = weighted_cube.sum(axis=(1, 2))
specaxis = cube.spectral_axis

error = rms*np.ones_like(beam_weighted_spectrum)
sp = pyspeckit.Spectrum(data=beam_weighted_spectrum, xarr=specaxis, error=error)
amplitude_guess = beam_weighted_spectrum.max().value
center_guess = ((beam_weighted_spectrum*specaxis).sum()/beam_weighted_spectrum.sum()).value
width_guess = 1.
guesses = [amplitude_guess, center_guess, width_guess]

fig = plt.figure(figsize=(4,4))
sp.specfit(fittype='gaussian', guesses=guesses)

sp.plotter()
sp.specfit.plot_fit()
# plt.savefig('C18O_beam_weighted_spectra_Per50.pdf', bbox_inches='tight')
