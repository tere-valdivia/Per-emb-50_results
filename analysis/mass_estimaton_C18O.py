import sys
sys.path.append('../')
import matplotlib.pyplot as plt
from uncertainties import umath
from uncertainties import ufloat, unumpy
from astropy.modeling.models import Gaussian2D
from astropy.coordinates import SkyCoord, FK5
import regions
import os
from NOEMAsetup import *
import numpy as np
import pandas as pd
from spectral_cube import SpectralCube
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u


'''
This code creates the N(C18O) maps and takes important statistics and quick
calculations.

Important functions moved to NOEMAsetup
'''

'''
Inputs
'''
# filenames
filenameC18O = '../' + C18O_2_1 + '_pbcor_reprojectH2COs_mom0_l'
filenameC18Okink = '../' + C18O_2_1 + '_pbcor_reprojectH2COs_mom0_l_kink'
tablefile_unumpy = 'M_H2_Tex_{0}_mom0_pbcor_unc_tmodel_{1}Msun.csv'
NC18Ofilename = 'column_dens_maps/N_C18O_constantTex_{0}K_mom0_pbcor.fits'
uNC18Ofilename = 'column_dens_maps/N_C18O_unc_constantTex_{0}K_mom0_pbcor.fits'
NC18Ofilenamekink = 'column_dens_maps/N_C18O_constantTex_{0}K_mom0_pbcor_kink.fits'
uNC18Ofilenamekink = 'column_dens_maps/N_C18O_unc_constantTex_{0}K_mom0_pbcor_kink.fits'

# constants
rms = 0.40993961429269554 * u.K * u.km/u.s
X_C18O = 5.9e6  # Frerking et al 1982
# this is the X_C18O value used in Nishimura et al 2015 for Orion clouds
distance = (dist_Per50 * u.pc).to(u.cm)
mu_H2 = 2.7
B0 = (54891.420 * u.MHz).to(1/u.s)
# In the future this can be changed to a T_ex map
Tex_u = ufloat(15., 5.)
M_env = 0.39 #* u.Msun
'''
End inputs
'''

NC18Oheader = fits.getheader(filenameC18O+'.fits')
restfreq = NC18Oheader['restfreq'] * u.Hz
primbeamFWHM = pb_noema(restfreq).to(u.deg).value
deltara = (NC18Oheader['CDELT1'] * u.deg).to(u.rad).value
deltadec = (NC18Oheader['CDELT2'] * u.deg).to(u.rad).value
wcsmom = WCS(NC18Oheader)
ra0_pix, dec0_pix = wcsmom.celestial.all_world2pix(ra_Per50.value, dec_Per50.value, 0)

# We obtain the Tdv and its shape to do a grid
mom0 = fits.getdata(filenameC18O+'.fits') * u.K * u.km/u.s
mom0kink = fits.getdata(filenameC18Okink+'.fits') * u.K * u.km/u.s
leny, lenx = np.shape(mom0)
yy, xx = np.mgrid[0:leny, 0:lenx]
# we need a map of the primary beam response to calculate the unc. properly
beamresponse = Gaussian2D(amplitude=1, x_mean=ra0_pix, y_mean=dec0_pix, x_stddev=primbeamFWHM/2.35/(
    deltadec*u.rad).to(u.deg).value, y_stddev=primbeamFWHM/2.35/(deltadec*u.rad).to(u.deg).value)(xx, yy)
assert np.shape(beamresponse) == np.shape(mom0)
u_mom0 = rms / beamresponse
u_mom0kink = np.where(np.isnan(mom0kink), np.nan, u_mom0)
u_mom0[np.where(np.isnan(mom0))] = np.nan * u_mom0.unit
mom0 = unumpy.uarray(mom0.value, u_mom0.value)  # they cannot be with units
mom0kink = unumpy.uarray(mom0kink.value, u_mom0kink.value)


# this is just to do a quick estimate of accretion timescale using t_ff
# M_s = 1.71 * u.Msun
M_s = ufloat(1.71, 0.19)

# M_env = ufloat(0.285, 0.105)
M_disk = 0.58 #* u.Msun
Mstar = (M_s + M_env + M_disk)

t_ff = t_freefall_unumpy(3300*u.AU, Mstar)

NC18Oheader['bunit'] = 'cm-2'

# Here we calculate one single map with T = 15pm5 K
formatname = str(int(Tex_u.n)) + 'pm' + str(int(Tex_u.s))

if os.path.exists('column_dens_maps/'+NC18Ofilename.format(formatname)):
    NC18O = fits.getdata('column_dens_maps/'+NC18Ofilename.format(formatname))
    uNC18O = fits.getdata('column_dens_maps/'+uNC18Ofilename.format(formatname))
    NC18O = unumpy.uarray(NC18O, uNC18O)
else:
    NC18O = N_C18O_21(mom0, B0, Tex_u)
    newfitshdu = fits.PrimaryHDU(data=unumpy.nominal_values(NC18O), header=NC18Oheader)
    newfitshdu.writeto('column_dens_maps/'+NC18Ofilename.format(formatname))
    newfitshdu = fits.PrimaryHDU(data=unumpy.std_devs(NC18O), header=NC18Oheader)
    newfitshdu.writeto('column_dens_maps/'+uNC18Ofilename.format(formatname))

if os.path.exists('column_dens_maps/'+NC18Ofilenamekink.format(formatname)):
    NC18Okink = fits.getdata('column_dens_maps/'+NC18Ofilenamekink.format(formatname))
    uNC18Okink = fits.getdata('column_dens_maps/'+uNC18Ofilenamekink.format(formatname))
    NC18Okink = unumpy.uarray(NC18Okink, uNC18Okink)
else:
    NC18Okink = N_C18O_21(mom0kink, B0, Tex_u)
    newfitshdu = fits.PrimaryHDU(data=unumpy.nominal_values(NC18Okink), header=NC18Oheader)
    newfitshdu.writeto('column_dens_maps/'+NC18Ofilenamekink.format(formatname))
    newfitshdu = fits.PrimaryHDU(data=unumpy.std_devs(NC18Okink), header=NC18Oheader)
    newfitshdu.writeto('column_dens_maps/'+uNC18Ofilenamekink.format(formatname))

# Now we take the total mass and estimate accretion
if os.path.exists(tablefile_unumpy.format(formatname, M_env)):
    results_mass_unumpy = pd.read_csv(tablefile_unumpy.format(formatname, M_env))
else:
    results_mass_unumpy = pd.DataFrame(index=['No kink', 'Kink'])
    for index, map in zip(results_mass_unumpy.index.values, [NC18O, NC18Okink]):
        results_mass_unumpy.loc[index, 'Mean NC18O (cm-2)'] = np.nanmean(map).n
        results_mass_unumpy.loc[index,
                                'Standard deviation NC18O (cm-2)'] = np.nanstd(unumpy.nominal_values(map))
        results_mass_unumpy.loc[index,
                                'Median NC18O (cm-2)'] = np.nanmedian(unumpy.nominal_values(map))
        results_mass_unumpy.loc[index, 'Min NC18O (cm-2)'] = np.nanmin(map).n
        results_mass_unumpy.loc[index, 'Max NC18O (cm-2)'] = np.nanmax(map).n
        NH2 = map * X_C18O
        NH2tot = np.nansum(NH2)
        print(NH2tot)
        MH2 = M_hydrogen2(NH2tot, mu_H2, distance, deltara, deltadec)
        results_mass_unumpy.loc[index, 'M (M_sun)'] = MH2.n
        results_mass_unumpy.loc[index, 'u M (M_sun)'] = MH2.s
        results_mass_unumpy.loc[index, 't_freefall (yr)'] = t_ff.n
        results_mass_unumpy.loc[index, 'u t_freefall (yr)'] = t_ff.s
        dotM = MH2/t_ff
        results_mass_unumpy.loc[index, 'dotM_in (M_sun/yr) (M_star=0.39Msun)'] = dotM.n
        results_mass_unumpy.loc[index, 'u dotM_in (M_sun/yr)'] = dotM.s
        Nsum = np.nansum(map)
        results_mass_unumpy.loc[index, 'Sum NC18O (cm-2 Npx)'] = Nsum.n
        results_mass_unumpy.loc[index, 'u Sum NC18O (cm-2 Npx)'] = Nsum.s

    results_mass_unumpy.to_csv(tablefile_unumpy.format(formatname, M_env))
print(results_mass_unumpy[['Sum NC18O (cm-2 Npx)', 'u Sum NC18O (cm-2 Npx)']])
