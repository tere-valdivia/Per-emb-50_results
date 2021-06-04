import numpy as np
import pandas as pd
from spectral_cube import SpectralCube
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import sys
sys.path.append('../')
from NOEMAsetup import *
import os
import regions
import matplotlib.pyplot as plt
from astropy.constants import G
from astropy.coordinates import SkyCoord, FK5
import aplpy
import copy
import velocity_tools.stream_lines as SL
import pickle
from uncertainties import ufloat, unumpy

# TODO: Add error propagation
'''
Important functions
'''

def t_freefall(r, M):
    '''
    r must be a AU Quantity
    M must be a M_sun Quantity
    Returns free-fall time in yr Quantity
    '''
    t= np.sqrt((r)**3/(G * M)).decompose().to(u.yr)
    return t

def distance_pix(x0, y0, x, y):
    dis = np.sqrt((x-x0)**2 + (y-y0)**2)
    return dis

def distance_physical(ra0, dec0, ra, dec, header):
    ra0_pix, dec0_pix = WCS(header).celestial.all_world2pix(ra0, dec0, 0)
    dist_pix = distance_pix(ra0_pix, dec0_pix, ra, dec)
    dist_deg = (np.abs(header['CDELT2'])* u.deg).to(u.arcsec) * dist_pix
    dist = dist_deg.value * dist_Per50 # pc * deg = au
    return dist

def M_hydrogen2(N, mu, D, deltara, deltadec):
    """
    Returns the gas mass of molecular hydrogen, given the sum of column density
    N in cm-2.

    Requires N to be in cm-2 but without unit. Returns in solar masses

    Args:
        variable (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """
    preamble = (mu * m_p) * (D**2) * np.abs(deltara * deltadec)
    Mass = N * preamble.value
    Mass_unit = ((1 * u.cm**(-2) * preamble.unit).to(u.Msun)).value
    # return Mass.to(u.Msun)
    return Mass * Mass_unit

def J_nu(nu, T):
    """
    Calculates the Rayleigh-Jeans equivalent temperature J_nu, in particular to
    aid in the calculation of the column density of C18O (but is used for any
    molecule)

    Note that the input parameters  must have their corresponding units

    Returns the equivalent temperature in u.K
    """
    over = h * nu / k_B
    under = np.exp(over/T) - 1
    return (over/under).decompose()

def Qrot(B0, Tex):
    """
    Calculates the partition function of a rigid rotor, diatomic molecule, with
    the 1st order Taylor approximation

    The partition function is adimensional,  so the function returns a float
    """
    taylorapp = k_B * Tex / (h * B0) + 1./3.
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
    '''
    nu = 219560.3541 * u.MHz
    constant = 3 * h / (8*np.pi**3 * (1.1079e-19 *u.esu *u.cm)**2 *2/5)
    Eu = 15.81 * u.K
    preamble = constant * Qrot(B0, Tex)/5 * np.exp(Eu / Tex) / (np.exp(10.54*u.K/Tex)-1) * 1/(J_nu(nu, Tex) - J_nu(nu, 2.73*u.K))
    NC18O = preamble.value * TdV/f
    N_unit = (1. * preamble.unit * u.K * u.km / u.s).to(u.cm**(-2)).value
    # return NC18O.to(u.cm**(-2))
    return NC18O * N_unit

# constant = 3 * h / (8*np.pi**3 * (1.1079e-19 *u.esu *u.cm)**2 *2/5)
# constant.decompose().to(u.s/u.km/u.cm**2)

'''
Inputs
'''
# filenameH2CO = '../H2CO/CDconfigsmall/Per-emb-50_CD_l021l060_uvsub_H2CO_multi_small_fitcube_fitted'
filenameC18O = '../' + C18O_2_1 + '_pbcor_reprojectH2COs_mom0_l'
filenameC18Okink = '../' + C18O_2_1 + '_pbcor_reprojectH2COs_mom0_l_kink'
tablefile = 'M_H2_Tex_fixed_mom0_pbcor_unc.csv'
tablefilekink = 'M_H2_Tex_fixed_mom0_pbcor_kink_unc.csv'
rms = 2.4041 * u.K * u.km/u.s
NC18Ofilename = 'N_C18O_constantTex_{0}K_mom0_pbcor.fits'
uNC18Ofilename = 'N_C18O_unc_constantTex_{0}K_mom0_pbcor.fits'
NC18Ofilenamekink = 'N_C18O_constantTex_{0}K_mom0_pbcor_kink.fits'
uNC18Ofilenamekink = 'N_C18O_unc_constantTex_{0}K_mom0_pbcor_kink.fits'
NC18Oplotname = 'N_C18O_constantTex_{0}K_mom0_pbcor.pdf'
X_C18O = 5.9e6 # Look for Frerking et al 1982
# this is the X_C18O value used in Nishimura et al 2015 for Orion clouds
distance = (dist_Per50 * u.pc).to(u.cm)
mu_H2 = 2.7
velinit = 5.5 * u.km/u.s
velend = 9.5 * u.km/u.s
Texlist = np.array([10,11,12,13,14,15])* u.K
B0 = (54891.420 * u.MHz).to(1/u.s)


'''
End inputs
'''

# reproject the C18O to the H2CO wcs
# if not os.path.exists(filenameC18O+'_reprojectH2COs_2.fits'):
#     cubeH2CO = SpectralCube.read(filenameH2CO+'.fits').with_spectral_unit(u.km/u.s).spectral_slab(velinit,velend)
#     cubeC18O = SpectralCube.read(filenameC18O+'.fits').with_spectral_unit(u.km/u.s)
#     spectral_grid_objective = cubeH2CO.spectral_axis
#     # spectral reprojection
#     cubeC18O = cubeC18O.spectral_interpolate(spectral_grid_objective)
#     # spatial reprojection
#     cubeC18O = cubeC18O.reproject(cubeH2CO.header) #beam is still there
#     cubeC18O.write(filenameC18O+'_reprojectH2COs_2.fits')
# else:
#     cubeC18O = SpectralCube.read(filenameC18O+'_reprojectH2COs_2.fits').with_spectral_unit(u.km/u.s)
#
# # Now leave out all that is not streamer
# if not os.path.exists(filenameC18O+'_H2COmasked_'+str(snratio)+'sigma_2.fits'):
#     if not 'cubeH2CO' in globals():
#         #  Just load if not loaded before
#         cubeH2CO = SpectralCube.read(filenameH2CO+'.fits').with_spectral_unit(u.km/u.s).spectral_slab(velinit,velend)
#     # mask the cube where there is emission
#     masked_cube = cubeC18O.with_mask(cubeH2CO > snratio* rms)
#     region_streamer = '../data/region_streamer_l.reg'
#     regio = regions.read_ds9(region_streamer)
#     streamer_cube = masked_cube.subcube_from_regions(regio)
#     streamer_cube.write(filenameC18O+'_H2COmasked_'+str(snratio)+'sigma_2.fits')
#
# streamer_cube = SpectralCube.read(filenameC18O+'_H2COmasked_'+str(snratio)+'sigma_2.fits')
# # We call it new because it is the header of the masked and cut C18O
# # newheaderC18O = cubeC18O.header
# newheaderC18O = streamer_cube.header
# deltara = (newheaderC18O['CDELT1'] * u.deg).to(u.rad).value
# deltadec = (newheaderC18O['CDELT2'] * u.deg).to(u.rad).value
#
# # Change cube to K
# k_streamer_cube = streamer_cube.to(u.K)
# # k_streamer_cube.write(filenameC18O+'_testK.fits')
#
# # do a moment 0
# # As it is the moment 0 of C18O where H2CO is positive,C18O can be negative
# mom0 = k_streamer_cube.moment(order=0)
# NC18Oheader = mom0.header
# mom0 = mom0.value * mom0.unit #Transform from type Projection to type Quantity
# mom0[np.where(mom0.value < 0.0)] = np.nan * u.K * u.km/u.s
# wcsmom = WCS(newheaderC18O).celestial
# # for now,lets assume a constant Tex
#
# NC18Oheader['bmaj'] = newheaderC18O['bmaj']
# NC18Oheader['bmin'] = newheaderC18O['bmin']
# NC18Oheader['bpa'] = newheaderC18O['bpa']
#
# #save the moment 0
# if not os.path.exists(filenameC18O+'_H2COmasked_'+str(snratio)+'sigma_mom0_2.fits'):
#     NC18Omomheader = NC18Oheader.copy()
#     NC18Omomheader['bunit'] = 'K km s-1'
#     newmom0hdu = fits.PrimaryHDU(data=mom0.value, header=NC18Omomheader)
#     newmom0hdu.writeto(filenameC18O+'_H2COmasked_'+str(snratio)+'sigma_mom0_2.fits')
NC18Oheader = fits.getheader(filenameC18O+'.fits')
deltara = (NC18Oheader['CDELT1'] * u.deg).to(u.rad).value
deltadec = (NC18Oheader['CDELT2'] * u.deg).to(u.rad).value
wcsmom = WCS(NC18Oheader)
mom0 = fits.getdata(filenameC18O+'.fits') *u.K * u.km/u.s
u_mom0 = rms * np.ones(np.shape(mom0))
mom0 = unumpy.uarray(mom0.value,u_mom0.value) # they cannot be with units
mom0kink = fits.getdata(filenameC18Okink+'.fits') *u.K * u.km/u.s
u_mom0kink = rms * np.ones(np.shape(mom0kink))
mom0kink = unumpy.uarray(mom0kink.value,u_mom0kink.value)

NC18Oheader['bunit'] = 'cm-2'

if os.path.exists(tablefilekink) and os.path.exists(tablefile):
    results_mass = pd.read_csv(tablefile)
    results_mass_kink = pd.read_csv(tablefilekink)
else:
    if not os.path.exists(tablefile):
        results_mass = pd.DataFrame(index=Texlist.value)

        for Tex in Texlist:
            # Do a N(C18O) map
            NC18O = N_C18O_21(mom0, B0, Tex) # the mom0 must have K km/s units
            #We save the column density obtained in a fits file
            if not os.path.exists('column_dens_maps/'+NC18Ofilename.format(Tex.value)):
                # newfitshdu = fits.PrimaryHDU(data=NC18O.value, header=NC18Oheader)
                newfitshdu = fits.PrimaryHDU(data=unumpy.nominal_values(NC18O), header=NC18Oheader)
                newfitshdu.writeto('column_dens_maps/'+NC18Ofilename.format(Tex.value))
            if not os.path.exists('column_dens_maps/'+uNC18Ofilename.format(Tex.value)):
                # newfitshdu = fits.PrimaryHDU(data=NC18O.value, header=NC18Oheader)
                newfitshdu = fits.PrimaryHDU(data=unumpy.std_devs(NC18O), header=NC18Oheader)
                newfitshdu.writeto('column_dens_maps/'+uNC18Ofilename.format(Tex.value))
            # We plot the column density
            # fig = plt.figure(figsize=(4,4))
            # ax = fig.add_subplot(111, projection=wcsmom)
            # im = ax.imshow(NC18O.value)
            # fig.colorbar(im,ax=ax,label=r'N(C$^{18}$O) (cm$^{-2}$)')
            # ax.set_xlabel('RA (J2000)')
            # ax.set_ylabel('DEC (J2000)')
            # if not os.path.exists('column_dens_maps/'+NC18Oplotname.format(Tex.value)):
            #     fig.savefig('column_dens_maps/'+NC18Oplotname.format(Tex.value))

            # Calculate statistics for future reference
            # results_mass.loc[Tex.value, 'Mean NC18O (cm-2)'] = np.nanmean(NC18O.value)
            results_mass.loc[Tex.value, 'Mean NC18O (cm-2)'] = np.nanmean(NC18O).n
            # results_mass.loc[Tex.value, 'Standard deviation NC18O (cm-2)'] = np.nanstd(NC18O.value)
            results_mass.loc[Tex.value, 'Standard deviation NC18O (cm-2)'] = np.nanstd(unumpy.nominal_values(NC18O))
            # results_mass.loc[Tex.value, 'Median NC18O (cm-2)'] = np.nanmedian(NC18O.value)
            results_mass.loc[Tex.value, 'Median NC18O (cm-2)'] = np.nanmedian(unumpy.nominal_values(NC18O))
            # results_mass.loc[Tex.value, 'Min NC18O (cm-2)'] = np.nanmin(NC18O.value)
            results_mass.loc[Tex.value, 'Min NC18O (cm-2)'] = np.nanmin(NC18O).n
            # results_mass.loc[Tex.value, 'Max NC18O (cm-2)'] = np.nanmax(NC18O.value)
            results_mass.loc[Tex.value, 'Max NC18O (cm-2)'] = np.nanmax(NC18O).n
            # Now, we calculate the column density of H2
            # results_mass.loc[Tex.value, 'Sum NC18O (cm-2 Npx)'] = NC18O.nansum().value
            Nsum = np.nansum(NC18O)
            results_mass.loc[Tex.value, 'Sum NC18O (cm-2 Npx)'] = Nsum.n
            results_mass.loc[Tex.value, 'u Sum NC18O (cm-2 Npx)'] = Nsum.s
            NH2 = NC18O * X_C18O
            NH2tot = np.nansum(NH2)
            # results_mass.loc[Tex.value, 'Sum NH2 (cm-2 Npx)'] = NH2tot.value
            results_mass.loc[Tex.value, 'Sum NH2 (cm-2 Npx)'] = NH2tot.n
            results_mass.loc[Tex.value, 'u Sum NH2 (cm-2 Npx)'] = NH2tot.s
            MH2 = M_hydrogen2(NH2tot, mu_H2, distance, deltara, deltadec)
            # MH2 = NH2tot * (mu_H2 * m_p) * (distance**2) * np.abs(deltara * deltadec)
            # results_mass.loc[Tex.value, 'M (kg)'] = (MH2.to(u.kg)).value
            # results_mass.loc[Tex.value, 'M (M_sun)'] = (MH2.to(u.Msun)).value
            results_mass.loc[Tex.value, 'M (M_sun)'] = MH2.n
            results_mass.loc[Tex.value, 'u M (M_sun)'] = MH2.s
            # print(Tex, MH2, MH2.to(u.Msun))

        results_mass.to_csv(tablefile)
    if not os.path.exists(tablefilekink):
        results_mass_kink = pd.DataFrame(index=Texlist.value)

        for Tex in Texlist:
            # Do a N(C18O) map
            NC18O = N_C18O_21(mom0kink, B0, Tex) # the mom0 must have K km/s units
            #We save the column density obtained in a fits file
            if not os.path.exists('column_dens_maps/'+NC18Ofilenamekink.format(Tex.value)):
                # newfitshdu = fits.PrimaryHDU(data=NC18O.value, header=NC18Oheader)
                newfitshdu = fits.PrimaryHDU(data=unumpy.nominal_values(NC18O), header=NC18Oheader)
                newfitshdu.writeto('column_dens_maps/'+NC18Ofilenamekink.format(Tex.value))
            if not os.path.exists('column_dens_maps/'+uNC18Ofilenamekink.format(Tex.value)):
                # newfitshdu = fits.PrimaryHDU(data=NC18O.value, header=NC18Oheader)
                newfitshdu = fits.PrimaryHDU(data=unumpy.std_devs(NC18O), header=NC18Oheader)
                newfitshdu.writeto('column_dens_maps/'+uNC18Ofilenamekink.format(Tex.value))
            # Calculate statistics for future reference
            # results_mass_kink.loc[Tex.value, 'Mean NC18O (cm-2)'] = np.nanmean(NC18O.value)
            results_mass_kink.loc[Tex.value, 'Mean NC18O (cm-2)'] = np.nanmean(NC18O).n
            # results_mass_kink.loc[Tex.value, 'Standard deviation NC18O (cm-2)'] = np.nanstd(NC18O.value)
            results_mass_kink.loc[Tex.value, 'Standard deviation NC18O (cm-2)'] = np.nanstd(unumpy.nominal_values(NC18O))
            # results_mass_kink.loc[Tex.value, 'Median NC18O (cm-2)'] = np.nanmedian(NC18O.value)
            results_mass_kink.loc[Tex.value, 'Median NC18O (cm-2)'] = np.nanmedian(unumpy.nominal_values(NC18O))
            # results_mass_kink.loc[Tex.value, 'Min NC18O (cm-2)'] = np.nanmin(NC18O.value)
            results_mass_kink.loc[Tex.value, 'Min NC18O (cm-2)'] = np.nanmin(NC18O).n
            # results_mass_kink.loc[Tex.value, 'Max NC18O (cm-2)'] = np.nanmax(NC18O.value)
            results_mass_kink.loc[Tex.value, 'Max NC18O (cm-2)'] = np.nanmax(NC18O).n
            # Now, we calculate the column density of H2
            # results_mass_kink.loc[Tex.value, 'Sum NC18O (cm-2 Npx)'] = NC18O.nansum().value
            Nsum = np.nansum(NC18O)
            results_mass_kink.loc[Tex.value, 'Sum NC18O (cm-2 Npx)'] = Nsum.n
            results_mass_kink.loc[Tex.value, 'u Sum NC18O (cm-2 Npx)'] = Nsum.s
            NH2 = NC18O * X_C18O
            NH2tot = np.nansum(NH2)
            # results_mass_kink.loc[Tex.value, 'Sum NH2 (cm-2 Npx)'] = NH2tot.value
            results_mass_kink.loc[Tex.value, 'Sum NH2 (cm-2 Npx)'] = NH2tot.n
            results_mass_kink.loc[Tex.value, 'u Sum NH2 (cm-2 Npx)'] = NH2tot.s
            MH2 = M_hydrogen2(NH2tot, mu_H2, distance, deltara, deltadec)
            # MH2 = NH2tot * (mu_H2 * m_p) * (distance**2) * np.abs(deltara * deltadec)
            # results_mass_kink.loc[Tex.value, 'M (kg)'] = (MH2.to(u.kg)).value
            # results_mass_kink.loc[Tex.value, 'M (M_sun)'] = (MH2.to(u.Msun)).value
            results_mass_kink.loc[Tex.value, 'M (M_sun)'] = MH2.n
            results_mass_kink.loc[Tex.value, 'u M (M_sun)'] = MH2.s
            # print(Tex, MH2, MH2.to(u.Msun))

        results_mass_kink.to_csv(tablefilekink)
    results_mass = pd.read_csv(tablefile)
    results_mass_kink = pd.read_csv(tablefilekink)

# NC18Omap = fits.getdata('column_dens_maps/'+NC18Ofilename.format(10.0))
NC18Omap = unumpy.uarray(fits.getdata('column_dens_maps/'+NC18Ofilename.format(10.0)), fits.getdata('column_dens_maps/'+uNC18Ofilename.format(10.0)))

# NC18Omapkink = fits.getdata('column_dens_maps/'+NC18Ofilenamekink.format(10.0))
NC18Omapkink = unumpy.uarray(fits.getdata('column_dens_maps/'+NC18Ofilenamekink.format(10.0)), fits.getdata('column_dens_maps/'+uNC18Ofilenamekink.format(10.0)))

wcsmap = WCS(fits.getheader('column_dens_maps/'+NC18Ofilename.format(10.0))).celestial

NH2map = NC18Omap * X_C18O # * (u.cm**-2)
NH2mapkink = NC18Omapkink * X_C18O # * (u.cm**-2)
leny, lenx = np.shape(NH2map)

# This is to have an idea of the total mass
M_s = 1.71*u.Msun
M_env = np.array([0.18,0.39])*u.Msun
M_disk = 0.58*u.Msun
Mstar = (M_s+M_env+M_disk)
t_ff = t_freefall(3300*u.AU, Mstar)
leny, lenx = np.shape(NC18Omap)

M_acc = unumpy.uarray(results_mass_kink['M (M_sun)'].values, results_mass_kink['u M (M_sun)'].values)
M_dot = [M_acc / t_ff[0].value, M_acc / t_ff[1].value]


# Now, we separate the streamer in bins
# Now we do the same for the streamer-calculated distances

xx, yy = np.meshgrid(range(lenx), range(leny))

distance_map = distance_physical(ra_Per50.value,dec_Per50.value, xx, yy, NC18Oheader)
binsize = 200 # au

modelname = 'H2CO_0.39Msun_env'
fileinpickle = 'streamer_model_'+modelname+'_params'
pickle_in = open(fileinpickle+'.pickle', "rb")
streamdict = pickle.load(pickle_in)
omega0 = streamdict['omega0']
r0 = streamdict['r0']
theta0 = streamdict['theta0']
phi0 = streamdict['phi0']
v_r0 = streamdict['v_r0']
inc = streamdict['inc']
PA_ang = streamdict['PA']
(x1, y1, z1), (vx1, vy1, vz1) = SL.xyz_stream(
    mass=Mstar[1], r0=r0, theta0=theta0, phi0=phi0,
    omega=omega0, v_r0=v_r0, inc=inc, pa=PA_ang, rmin=10*u.au)
rc = SL.r_cent(mass=Mstar[1], omega=omega0, r0=r0)

# y1 is in au
dist_streamer = np.sqrt(x1**2+y1**2+z1**2)
dist_projected = np.sqrt(x1**2+z1**2)


radiuses = np.arange(0,3300, binsize) # list of streamer lengths we want to sample
binradii = np.arange(100,3300,binsize) # u.AU  of  the streamer length
masses = unumpy.uarray(np.zeros(len(binradii)), np.zeros(len(binradii)))
masseskink = unumpy.uarray(np.zeros(len(binradii)), np.zeros(len(binradii)))
# masses = np.zeros(len(binradii)) * u.Msun
# masseskink = np.zeros(len(binradii)) * u.Msun
times = np.zeros((len(binradii),2)) * u.yr
# m_acclist = np.zeros((len(binradii),2)) * u.Msun / u.yr
# m_acclistkink = np.zeros((len(binradii),2)) * u.Msun / u.yr
m_acclist = unumpy.uarray(np.zeros((len(binradii),2)), np.zeros((len(binradii),2)))
m_acclistkink = unumpy.uarray(np.zeros((len(binradii),2)), np.zeros((len(binradii),2)))

for i in range(len(radiuses)-1):
    xzrange = dist_projected[np.where((dist_streamer.value>radiuses[i]) & (dist_streamer.value<radiuses[i+1]))]
    mask = np.where((distance_map>np.amin(xzrange.value)) & (distance_map<np.amax(xzrange.value)))
    # mask = np.where((distance_map>radiuses[i]) & (distance_map<radiuses[i+1]))
    NH2tot = np.nansum(NH2map[mask])
    NH2totkink = np.nansum(NH2mapkink[mask])
    masses[i] = M_hydrogen2(NH2tot, mu_H2, distance, deltara, deltadec)
    masseskink[i] = M_hydrogen2(NH2totkink, mu_H2, distance, deltara, deltadec)
    times[i] = t_freefall(binradii[i]*u.AU, Mstar)
    m_acclist[i] = [masses[i] / times[i,0].value, masses[i] / times[i,1].value]
    m_acclistkink[i] = [masseskink[i] / times[i,0].value, masseskink[i] / times[i,1].value]

fig = plt.figure(figsize=(6,6))
ax3 = fig.add_subplot(313)
ax3.errorbar(binradii[1:], unumpy.nominal_values(m_acclist[1:,1]), yerr=unumpy.std_devs(m_acclist[1:,1]), marker='o', linestyle='none', mfc='none', mec='r', ecolor='r')
ax3.errorbar(binradii[1:], unumpy.nominal_values(m_acclistkink[1:,1]), yerr=unumpy.std_devs(m_acclistkink[1:,1]), marker='o', linestyle='none', mfc='r', mec='r', ecolor='r')
# ax3.scatter(binradii[1:], m_acclistkink[1:,1].value, s=50, facecolors='r', edgecolors='r')
ax3.set_yscale('log')
ax3.set_xlabel('Distance from Protostar (au)')
ax3.set_ylabel(r'$\dot{M}$ (M$_{\odot}$ yr$^{-1}$)')
ax3.annotate('bin size = {} au'.format(binsize), (0.6,0.8), xycoords='axes fraction', size=14)
ax3.axvline(293, color='k', linestyle='dashed')
ax3.axvline(rc.value, color='b', linestyle='dotted')
ax3.set_xlim([0,3300])

ax = fig.add_subplot(311, sharex=ax3)
ax.errorbar(binradii[1:], unumpy.nominal_values(masses[1:])*1e3, yerr=unumpy.std_devs(masses[1:])*1e3, marker='o', linestyle='none', mfc='none', mec='r', ecolor='r', label='No kink')
ax.errorbar(binradii[1:], unumpy.nominal_values(masseskink[1:])*1e3, yerr=unumpy.std_devs(masseskink[1:])*1e3, marker='o', linestyle='none', mfc='r', mec='r', ecolor='r', label='kink')
# ax.scatter(binradii[1:], masses[1:].value*1e3, s=50, facecolors='none', edgecolors='r', label='No kink')
# ax.scatter(binradii[1:], masseskink[1:].value*1e3, s=50, facecolors='r', edgecolors='r',label='kink')
ax.set_ylabel(r'Mass in bin ($\times 10^3$ M$_{\odot}$)')
ax.legend(loc=4)
ax.axvline(293, color='k', linestyle='dashed')
# ax.annotate('Beam FWHM',(0.12,0.8), xycoords='axes fraction', size=14, rotation=90)
ax.text(350,0.2, 'Beam FWHM', size=10, rotation=90)
ax.axvline(rc.value, color='b', linestyle='dotted')
ax.text(150,0.2, r'$r_c=$'+str(round(rc.value,1))+' au', size=10, rotation=90, color='b')

ax2 = fig.add_subplot(312, sharex=ax3)
ax2.scatter(binradii[1:], times[1:,1].value, s=50, facecolors='b', edgecolors='b', label=r'$M_{*}=$'+str(Mstar[1]))
ax2.set_ylabel(r'Free-fall timescale (yr$^{-1}$)')
ax2.set_yscale('log')
ax2.legend()
ax2.axvline(293, color='k', linestyle='dashed')
ax2.axvline(rc.value, color='b', linestyle='dotted')
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax.get_xticklabels(), visible=False)

# fig.savefig('column_dens_maps/plot_mass_accretion_radius.pdf', dpi=300, bbox_inches='tight')

def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s} %"

# fig2 = plt.figure(figsize=(4,4))
# ax = fig2.add_subplot(111, projection=wcsmap)
# cmap = copy.copy(plt.cm.viridis)
# cmap.set_bad(np.array((1,1,1))*0.85)

fig3 = plt.figure(figsize=(4,4))
gc = aplpy.FITSFigure('column_dens_maps/'+NC18Ofilenamekink.format(10.0), figure=fig3)
gc.show_colorscale()
gc.add_colorbar()
gc.add_beam()
gc.beam.set_color('k')
gc.beam.set_frame(False)
gc.set_nan_color(np.array((1,1,1))*0.85)
gc.colorbar.set_axis_label_text(r'N(C$^{18}$O) (cm$^{-2}$)')
# fig3.savefig('column_dens_maps/N_C18O_constantTex_{0}K_mom0_pbcor_kink_aplpy.pdf'.format(10.0), dpi=300, bbox_inches='tight')

fig4 = plt.figure(figsize=(4,4))
ax4 = fig4.add_subplot(111)
ax4.plot(dist_projected.value,dist_streamer.value-dist_projected.value,'g-')
ax4.axvline(rc.value, color='b', linestyle='dotted', label=r'$r_c=$'+str(round(rc.value,1))+' au')
ax4.axvline(293, color='k', linestyle='dashed', label='Beam FWHM')
ax4.set_xlabel('Projected distance (au)')
ax4.set_ylabel(r'Difference b/w 3D and projected distance (au)')
ax4.legend()
# fig4.savefig(fileinpickle+'_distance_projection.pdf', dpi=300, bbox_inches='tight')

# If we want to know the angle

# angles = (np.arccos(dist_projected.value/dist_streamer.value) * u.rad).to(u.deg).value
# plt.plot(dist_projected.value,y1.value)
