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

Important functions
'''

# M_hydrogen2 moved to NOEMAsetup


def J_nu(nu, T):
    """
    Calculates the Rayleigh-Jeans equivalent temperature J_nu, in particular to
    aid in the calculation of the column density of C18O (but is used for any
    molecule)

    Note that the input parameter nu must have its corresponding unit

    Returns the equivalent temperature in K but no quantity
    changed the np.exp to umath.exp
    """
    over = (h * nu / k_B).to(u.K)
    under = umath.exp(over.value/T) - 1.
    return (over.value/under)


def Qrot(B0, Tex):
    """
    Calculates the partition function of a rigid rotor, diatomic molecule, with
    the 1st order Taylor approximation

    The partition function is adimensional,  so the function returns a float

    Tex must not be a quantity if it has an associated error
    """
    preamble = (k_B / (h * B0)).to(1/u.K)
    taylorapp = preamble.value * Tex + 1./3.
    return taylorapp


def N_C18O_21(TdV, B0, Tex, f=1):
    '''
    Returns the column density of C18O based on the J=2-1 transition

    To check if the constant is ok, I calculated the constant for J=1-0,
    obtained the same as for the example in Mangum + Shirrley 2015 and applied
    the same method with the different values of Eu, J and nu

    This is equivalent to combining equations 10 and 12 from Nishimura et al
    2015 using the optically thin limit (checked). The constant of equation 10
    is the same constant we get here divided by k_b/hB0 and multiplied by
    2J+1 = 5

    TdVin must be in K km/s, but must not be a Quantity, but a ufloat
    Tex must be in K, but must not be a Quantity, but a ufloat

    The returned column density has units of cm-2 but is no Quantity,
    so that it can be used with the uncertainties package
    changed the np.exp to umath.exp
    '''
    nu = 219560.3541 * u.MHz
    constant = 3. * h / (8.*np.pi**3 * (1.1079e-19 * u.esu * u.cm)**2 * 2./5.)
    Eu = 15.81  # K
    preamble = constant.value * Qrot(B0, Tex)/5. * umath.exp(Eu/Tex) / \
        (umath.exp(10.54/Tex) - 1.) * 1. / (J_nu(nu, Tex) - J_nu(nu, 2.73))
    NC18O = preamble * TdV / f
    N_unit = (1. * constant.unit * u.km / u.s).to(u.cm**(-2)).value
    return NC18O * N_unit


'''
Inputs
'''
# filenames
filenameC18O = '../' + C18O_2_1 + '_pbcor_reprojectH2COs_mom0_l'
filenameC18Okink = '../' + C18O_2_1 + '_pbcor_reprojectH2COs_mom0_l_kink'
tablefile = 'M_H2_Tex_fixed_mom0_pbcor_unc.csv'
tablefilekink = 'M_H2_Tex_fixed_mom0_pbcor_kink_unc.csv'
tablefile_unumpy = 'M_H2_Tex_{0}_mom0_pbcor_unc.csv'
NC18Ofilename = 'N_C18O_constantTex_{0}K_mom0_pbcor.fits'
uNC18Ofilename = 'N_C18O_unc_constantTex_{0}K_mom0_pbcor.fits'
NC18Ofilenamekink = 'N_C18O_constantTex_{0}K_mom0_pbcor_kink.fits'
uNC18Ofilenamekink = 'N_C18O_unc_constantTex_{0}K_mom0_pbcor_kink.fits'
NC18Oplotname = 'N_C18O_constantTex_{0}K_mom0_pbcor.pdf'
# constants
rms = 0.223 * u.K * u.km/u.s
X_C18O = 5.9e6  # Frerking et al 1982
# this is the X_C18O value used in Nishimura et al 2015 for Orion clouds
distance = (dist_Per50 * u.pc).to(u.cm)
mu_H2 = 2.7
B0 = (54891.420 * u.MHz).to(1/u.s)
# You can use an array of temps or a ufloat
Texlist = np.array([10, 11, 12, 13, 14, 15]) * u.K
Tex_u = ufloat(15, 5)
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
leny, lenx = np.shape(mom0)
yy, xx = np.mgrid[0:leny, 0:lenx]
# we need a map of the primary beam response to calculate the unc. properly
beamresponse = Gaussian2D(amplitude=1, x_mean=ra0_pix, y_mean=dec0_pix, x_stddev=primbeamFWHM/2.35/(
    deltadec*u.rad).to(u.deg).value, y_stddev=primbeamFWHM/2.35/(deltadec*u.rad).to(u.deg).value)(xx, yy)
assert np.shape(beamresponse) == np.shape(mom0)
u_mom0 = rms / beamresponse
mom0 = unumpy.uarray(mom0.value, u_mom0.value)  # they cannot be with units
mom0kink = fits.getdata(filenameC18Okink+'.fits') * u.K * u.km/u.s
mom0kink = unumpy.uarray(mom0kink.value, u_mom0.value)

# this time is just to do a quick estimate of accretion timescale using t_ff
M_s = 1.71*u.Msun
M_env = np.array([0.18, 0.39])*u.Msun
M_disk = 0.58*u.Msun
Mstar = (M_s+M_env+M_disk)
t_ff = t_freefall(3300*u.AU, Mstar)

NC18Oheader['bunit'] = 'cm-2'

# Here we calculate one single map with T = 15pm5 K
formatname = str(int(Tex_u.n)) + 'pm' + str(int(Tex_u.s))

NC18O = N_C18O_21(mom0, B0, Tex_u)

if not os.path.exists('column_dens_maps/'+NC18Ofilename.format(formatname)):
    newfitshdu = fits.PrimaryHDU(data=unumpy.nominal_values(NC18O), header=NC18Oheader)
    newfitshdu.writeto('column_dens_maps/'+NC18Ofilename.format(formatname))
if not os.path.exists('column_dens_maps/'+uNC18Ofilename.format(formatname)):
    newfitshdu = fits.PrimaryHDU(data=unumpy.std_devs(NC18O), header=NC18Oheader)
    newfitshdu.writeto('column_dens_maps/'+uNC18Ofilename.format(formatname))

NC18Okink = N_C18O_21(mom0kink, B0, Tex_u)
if not os.path.exists('column_dens_maps/'+NC18Ofilenamekink.format(formatname)):
    newfitshdu = fits.PrimaryHDU(data=unumpy.nominal_values(NC18Okink), header=NC18Oheader)
    newfitshdu.writeto('column_dens_maps/'+NC18Ofilenamekink.format(formatname))
if not os.path.exists('column_dens_maps/'+uNC18Ofilenamekink.format(formatname)):
    newfitshdu = fits.PrimaryHDU(data=unumpy.std_devs(NC18Okink), header=NC18Oheader)
    newfitshdu.writeto('column_dens_maps/'+uNC18Ofilenamekink.format(formatname))

# Now we take the total mass and estimate accretion
if os.path.exists(tablefile_unumpy.format(formatname)):
    results_mass_unumpy = pd.read_csv(tablefile_unumpy.format(formatname))
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
        MH2 = M_hydrogen2(NH2tot, mu_H2, distance, deltara, deltadec)
        results_mass_unumpy.loc[index, 'M (M_sun)'] = MH2.n
        results_mass_unumpy.loc[index, 'u M (M_sun)'] = MH2.s
        results_mass_unumpy.loc[index, 'M_acc (M_sun/yr) (M_star=0.39Msun)'] = MH2.n/t_ff[1].value
        results_mass_unumpy.loc[index, 'u M_acc (M_sun/yr)'] = MH2.s/t_ff[1].value
        Nsum = np.nansum(map)
        results_mass_unumpy.loc[index, 'Sum NC18O (cm-2 Npx)'] = Nsum.n
        results_mass_unumpy.loc[index, 'u Sum NC18O (cm-2 Npx)'] = Nsum.s

    results_mass_unumpy.to_csv(tablefile_unumpy.format(formatname))

# change this part of the code to work with the new functions
# Here we calculate the NC18O map and statistics with Tex with no errors

# if os.path.exists(tablefilekink) and os.path.exists(tablefile):
#     results_mass = pd.read_csv(tablefile)
#     results_mass_kink = pd.read_csv(tablefilekink)
# else:
#     if not os.path.exists(tablefile):
#         results_mass = pd.DataFrame(index=Texlist.value)
#         ## TODO: Move this to another file
#         for Tex in Texlist:
#             # Do a N(C18O) map
#             # the mom0 must have K km/s units but no Quantity
#             NC18O = N_C18O_21(mom0, B0, Tex)
#             #We save the column density obtained in a fits file
#             if not os.path.exists('column_dens_maps/'+NC18Ofilename.format(Tex.value)):
#                 # newfitshdu = fits.PrimaryHDU(data=NC18O.value, header=NC18Oheader)
#                 newfitshdu = fits.PrimaryHDU(data=unumpy.nominal_values(NC18O), header=NC18Oheader)
#                 newfitshdu.writeto('column_dens_maps/'+NC18Ofilename.format(Tex.value))
#             if not os.path.exists('column_dens_maps/'+uNC18Ofilename.format(Tex.value)):
#                 # newfitshdu = fits.PrimaryHDU(data=NC18O.value, header=NC18Oheader)
#                 newfitshdu = fits.PrimaryHDU(data=unumpy.std_devs(NC18O), header=NC18Oheader)
#                 newfitshdu.writeto('column_dens_maps/'+uNC18Ofilename.format(Tex.value))
#             # We plot the column density
#             # fig = plt.figure(figsize=(4,4))
#             # ax = fig.add_subplot(111, projection=wcsmom)
#             # im = ax.imshow(NC18O.value)
#             # fig.colorbar(im,ax=ax,label=r'N(C$^{18}$O) (cm$^{-2}$)')
#             # ax.set_xlabel('RA (J2000)')
#             # ax.set_ylabel('DEC (J2000)')
#             # if not os.path.exists('column_dens_maps/'+NC18Oplotname.format(Tex.value)):
#             #     fig.savefig('column_dens_maps/'+NC18Oplotname.format(Tex.value))
#
#             # Calculate statistics for future reference
#             # results_mass.loc[Tex.value, 'Mean NC18O (cm-2)'] = np.nanmean(NC18O.value)
#             results_mass.loc[Tex.value, 'Mean NC18O (cm-2)'] = np.nanmean(NC18O).n
#             # results_mass.loc[Tex.value, 'Standard deviation NC18O (cm-2)'] = np.nanstd(NC18O.value)
#             results_mass.loc[Tex.value, 'Standard deviation NC18O (cm-2)'] = np.nanstd(unumpy.nominal_values(NC18O))
#             # results_mass.loc[Tex.value, 'Median NC18O (cm-2)'] = np.nanmedian(NC18O.value)
#             results_mass.loc[Tex.value, 'Median NC18O (cm-2)'] = np.nanmedian(unumpy.nominal_values(NC18O))
#             # results_mass.loc[Tex.value, 'Min NC18O (cm-2)'] = np.nanmin(NC18O.value)
#             results_mass.loc[Tex.value, 'Min NC18O (cm-2)'] = np.nanmin(NC18O).n
#             # results_mass.loc[Tex.value, 'Max NC18O (cm-2)'] = np.nanmax(NC18O.value)
#             results_mass.loc[Tex.value, 'Max NC18O (cm-2)'] = np.nanmax(NC18O).n
#             # Now, we calculate the column density of H2
#             # results_mass.loc[Tex.value, 'Sum NC18O (cm-2 Npx)'] = NC18O.nansum().value
#             Nsum = np.nansum(NC18O)
#             results_mass.loc[Tex.value, 'Sum NC18O (cm-2 Npx)'] = Nsum.n
#             results_mass.loc[Tex.value, 'u Sum NC18O (cm-2 Npx)'] = Nsum.s
#             NH2 = NC18O * X_C18O
#             NH2tot = np.nansum(NH2)
#             # results_mass.loc[Tex.value, 'Sum NH2 (cm-2 Npx)'] = NH2tot.value
#             results_mass.loc[Tex.value, 'Sum NH2 (cm-2 Npx)'] = NH2tot.n
#             results_mass.loc[Tex.value, 'u Sum NH2 (cm-2 Npx)'] = NH2tot.s
#             MH2 = M_hydrogen2(NH2tot, mu_H2, distance, deltara, deltadec)
#             # MH2 = NH2tot * (mu_H2 * m_p) * (distance**2) * np.abs(deltara * deltadec)
#             # results_mass.loc[Tex.value, 'M (kg)'] = (MH2.to(u.kg)).value
#             # results_mass.loc[Tex.value, 'M (M_sun)'] = (MH2.to(u.Msun)).value
#             results_mass.loc[Tex.value, 'M (M_sun)'] = MH2.n
#             results_mass.loc[Tex.value, 'u M (M_sun)'] = MH2.s
#             results_mass.loc[Tex.value, 'M_acc (M_sun/yr) (M_star=0.39Msun)'] = MH2.n/t_ff[1].value
#             results_mass.loc[Tex.value, 'u M_acc (M_sun/yr)'] = MH2.s/t_ff[1].value
#             # print(Tex, MH2, MH2.to(u.Msun))
#
#         results_mass.to_csv(tablefile)
#     if not os.path.exists(tablefilekink):
#         results_mass_kink = pd.DataFrame(index=Texlist.value)
#
#         for Tex in Texlist:
#             # Do a N(C18O) map
#             NC18O = N_C18O_21(mom0kink, B0, Tex) # the mom0 must have K km/s units
#             #We save the column density obtained in a fits file
#             if not os.path.exists('column_dens_maps/'+NC18Ofilenamekink.format(Tex.value)):
#                 # newfitshdu = fits.PrimaryHDU(data=NC18O.value, header=NC18Oheader)
#                 newfitshdu = fits.PrimaryHDU(data=unumpy.nominal_values(NC18O), header=NC18Oheader)
#                 newfitshdu.writeto('column_dens_maps/'+NC18Ofilenamekink.format(Tex.value))
#             if not os.path.exists('column_dens_maps/'+uNC18Ofilenamekink.format(Tex.value)):
#                 # newfitshdu = fits.PrimaryHDU(data=NC18O.value, header=NC18Oheader)
#                 newfitshdu = fits.PrimaryHDU(data=unumpy.std_devs(NC18O), header=NC18Oheader)
#                 newfitshdu.writeto('column_dens_maps/'+uNC18Ofilenamekink.format(Tex.value))
#             # Calculate statistics for future reference
#             # results_mass_kink.loc[Tex.value, 'Mean NC18O (cm-2)'] = np.nanmean(NC18O.value)
#             results_mass_kink.loc[Tex.value, 'Mean NC18O (cm-2)'] = np.nanmean(NC18O).n
#             # results_mass_kink.loc[Tex.value, 'Standard deviation NC18O (cm-2)'] = np.nanstd(NC18O.value)
#             results_mass_kink.loc[Tex.value, 'Standard deviation NC18O (cm-2)'] = np.nanstd(unumpy.nominal_values(NC18O))
#             # results_mass_kink.loc[Tex.value, 'Median NC18O (cm-2)'] = np.nanmedian(NC18O.value)
#             results_mass_kink.loc[Tex.value, 'Median NC18O (cm-2)'] = np.nanmedian(unumpy.nominal_values(NC18O))
#             # results_mass_kink.loc[Tex.value, 'Min NC18O (cm-2)'] = np.nanmin(NC18O.value)
#             results_mass_kink.loc[Tex.value, 'Min NC18O (cm-2)'] = np.nanmin(NC18O).n
#             # results_mass_kink.loc[Tex.value, 'Max NC18O (cm-2)'] = np.nanmax(NC18O.value)
#             results_mass_kink.loc[Tex.value, 'Max NC18O (cm-2)'] = np.nanmax(NC18O).n
#             # Now, we calculate the column density of H2
#             # results_mass_kink.loc[Tex.value, 'Sum NC18O (cm-2 Npx)'] = NC18O.nansum().value
#             Nsum = np.nansum(NC18O)
#             results_mass_kink.loc[Tex.value, 'Sum NC18O (cm-2 Npx)'] = Nsum.n
#             results_mass_kink.loc[Tex.value, 'u Sum NC18O (cm-2 Npx)'] = Nsum.s
#             NH2 = NC18O * X_C18O
#             NH2tot = np.nansum(NH2)
#             # results_mass_kink.loc[Tex.value, 'Sum NH2 (cm-2 Npx)'] = NH2tot.value
#             results_mass_kink.loc[Tex.value, 'Sum NH2 (cm-2 Npx)'] = NH2tot.n
#             results_mass_kink.loc[Tex.value, 'u Sum NH2 (cm-2 Npx)'] = NH2tot.s
#             MH2 = M_hydrogen2(NH2tot, mu_H2, distance, deltara, deltadec)
#             # MH2 = NH2tot * (mu_H2 * m_p) * (distance**2) * np.abs(deltara * deltadec)
#             # results_mass_kink.loc[Tex.value, 'M (kg)'] = (MH2.to(u.kg)).value
#             # results_mass_kink.loc[Tex.value, 'M (M_sun)'] = (MH2.to(u.Msun)).value
#             results_mass_kink.loc[Tex.value, 'M (M_sun)'] = MH2.n
#             results_mass_kink.loc[Tex.value, 'u M (M_sun)'] = MH2.s
#             results_mass_kink.loc[Tex.value, 'M_acc (M_sun/yr) (M_star=0.39Msun)'] = MH2.n/t_ff[1].value
#             results_mass_kink.loc[Tex.value, 'u M_acc (M_sun/yr)'] = MH2.s/t_ff[1].value
#             # print(Tex, MH2, MH2.to(u.Msun))
#
#         results_mass_kink.to_csv(tablefilekink)
#     results_mass = pd.read_csv(tablefile)
#     results_mass_kink = pd.read_csv(tablefilekink)
