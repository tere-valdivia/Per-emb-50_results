import numpy as np
import pandas as pd
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
NC18Ofilename = 'N_C18O_constantTex_{0}K.fits'
NC18Oplotname = 'N_C18O_constantTex_{0}K.pdf'
X_C18O = 5.9e6 # Look for Frerking et al 1982
# this is the X_C18O value used in Nishimura et al 2015 for Orion clouds
distance = (dist_Per50 * u.pc).to(u.cm)
mu_H2 = 2.7

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
deltara = (newheaderC18O['CDELT1'] * u.deg).to(u.rad).value
deltadec = (newheaderC18O['CDELT2'] * u.deg).to(u.rad).value

# Change cube to K
k_streamer_cube = streamer_cube.to(u.K)

# do a moment 1
mom0 = k_streamer_cube.moment(order=0)
wcsmom = WCS(newheaderC18O).celestial

# Do a N(C18O) map
# for now,lets assume a constant Tex
# Tex = 10 * u.K
Texlist = np.array([10,11,12,13,14,15])* u.K
B0 = (54891.420 * u.MHz).to(1/u.s)
NC18Oheader = wcsmom.to_header()
NC18Oheader['bmaj'] = newheaderC18O['bmaj']
NC18Oheader['bmin'] = newheaderC18O['bmin']
NC18Oheader['bpa'] = newheaderC18O['bpa']
NC18Oheader['bunit'] = 'cm-2'

results_mass = pd.DataFrame(columns=['Sum (cm-2 Npx)','M (M_sun)'], index=Texlist.value)

for Tex in Texlist:
    NC18O = N_C18O_21(mom0, B0, Tex)
    #We save the column density obtained in a fits file
    if not os.path.exists(NC18Ofilename.format(Tex.value)):
        newfitshdu = fits.PrimaryHDU(data=NC18O.value, header=NC18Oheader)
        newfitshdu.writeto(NC18Ofilename.format(Tex.value))
    # We plot the column density
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection=wcsmom)
    im = ax.imshow(NC18O.value)
    fig.colorbar(im,ax=ax,label=r'N(C$^{18}$O) (cm$^{-2}$)')
    if not os.path.exists(NC18Oplotname.format(Tex.value)):
        fig.savefig(NC18Oplotname.format(Tex.value))

    # Now, we calculate the column density of H2
    results_mass.loc[Tex.value, 'Sum (cm-2 Npx)'] = NC18O.nansum().value
    NH2 = NC18O * X_C18O
    NH2tot = NH2.nansum()
    results_mass.loc[Tex.value, 'Sum NH2 (cm-2 Npx)'] = NH2tot.value
    MH2 = NH2tot * (mu_H2 * m_p) * (distance**2) * np.abs(deltara * deltadec)
    results_mass.loc[Tex.value, 'M (M_sun)'] = ((MH2.value)*u.kg).to(u.Msun).value

results_mass.to_csv('M_H2_Tex_fixed.csv')
