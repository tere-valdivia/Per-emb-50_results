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
from astropy.modeling.models import Gaussian2D
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
tablefilemacc = 'M_Mdot_Tex_fixed_mom0_pbcor_binned_unc.csv'
rms = 2.4041 * u.K * u.km/u.s
NC18Ofilename = 'N_C18O_constantTex_{0}K_mom0_pbcor.fits'
uNC18Ofilename = 'N_C18O_unc_constantTex_{0}K_mom0_pbcor.fits'
NC18Ofilenamekink = 'N_C18O_constantTex_{0}K_mom0_pbcor_kink.fits'
uNC18Ofilenamekink = 'N_C18O_unc_constantTex_{0}K_mom0_pbcor_kink.fits'
NC18Oplotname = 'N_C18O_constantTex_{0}K_mom0_pbcor.pdf'
radius3Dmapname = 'column_dens_maps/distance_3D_map.fits'
radiusmapname = 'column_dens_maps/distance_2D_map.fits'
X_C18O = 5.9e6 # Look for Frerking et al 1982
# this is the X_C18O value used in Nishimura et al 2015 for Orion clouds
distance = (dist_Per50 * u.pc).to(u.cm)
mu_H2 = 2.7
velinit = 5.5 * u.km/u.s
velend = 9.5 * u.km/u.s
Texlist = np.array([10,11,12,13,14,15])* u.K
B0 = (54891.420 * u.MHz).to(1/u.s)
binsize = 360 # au

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
# we need a map of the primary beam response to calculate the unc. properly
mom0 = fits.getdata(filenameC18O+'.fits') *u.K * u.km/u.s
leny, lenx = np.shape(mom0)
yy, xx = np.mgrid[0:leny, 0:lenx]
beamresponse = Gaussian2D(amplitude=1, x_mean=ra0_pix, y_mean=dec0_pix, x_stddev=primbeamFWHM/2.35/(deltadec*u.rad).to(u.deg).value,y_stddev=primbeamFWHM/2.35/(deltadec*u.rad).to(u.deg).value)(xx,yy)
# u_mom0 = rms * np.ones(np.shape(mom0))
u_mom0 = rms / beamresponse
mom0 = unumpy.uarray(mom0.value,u_mom0.value) # they cannot be with units
mom0kink = fits.getdata(filenameC18Okink+'.fits') *u.K * u.km/u.s
# u_mom0kink = rms * np.ones(np.shape(mom0kink))
u_mom0kink = rms / beamresponse
mom0kink = unumpy.uarray(mom0kink.value,u_mom0kink.value)

NC18Oheader['bunit'] = 'cm-2'

# Here we calculate the NC18O map and statistics
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

distance_map = distance_physical(ra_Per50.value,dec_Per50.value, xx, yy, NC18Oheader)


# We load the model
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
mass_streamer_table = pd.DataFrame()
dist_streamer = np.sqrt(x1**2+y1**2+z1**2)
dist_projected = np.sqrt(x1**2+z1**2)

radiuses = np.arange(0.,3200., binsize) # list of projected lengths we want to sample
# binradii = np.arange(binsize/2,3300.,binsize) # u.AU  of the streamer length
binradii = np.zeros(len(radiuses)-1) * u.AU
masses = unumpy.uarray(np.zeros(len(binradii)), np.zeros(len(binradii)))
masseskink = unumpy.uarray(np.zeros(len(binradii)), np.zeros(len(binradii)))
# masses = np.zeros(len(binradii)) * u.Msun
# masseskink = np.zeros(len(binradii)) * u.Msun
times = np.zeros((len(binradii),2)) * u.yr
# m_acclist = np.zeros((len(binradii),2)) * u.Msun / u.yr
# m_acclistkink = np.zeros((len(binradii),2)) * u.Msun / u.yr
m_acclist = unumpy.uarray(np.zeros(len(binradii)), np.zeros(len(binradii)))
m_acclistkink = unumpy.uarray(np.zeros(len(binradii)), np.zeros(len(binradii)))
radius3Dmap = np.zeros(np.shape(mom0)) * np.nan
# We need to save the previous limit
# aux_min = 0.

mass_streamer_table['2D bin minimum (au)'] = radiuses[:len(radiuses)-1]
mass_streamer_table['2D bin maximum (au)'] = radiuses[1:]

for i in range(len(radiuses)-1):
    # we want to make a map of the bins, so we need to save the indexes
    mask = np.where((distance_map>radiuses[i]) & (distance_map<radiuses[i+1]))
    # This is the mask of the pixels in the map
    # within these pixels are the x,z of the streamer
    # streamer distances within these bin of projected distances
    streamer_distances = dist_streamer[np.where((dist_projected.value>radiuses[i]) & (dist_projected.value<radiuses[i+1]))]
    # Now, what is the mean 3D distance for this bin
    binradii[i] = np.mean(streamer_distances)
    # xzrange = dist_projected[np.where((dist_streamer.value>radiuses[i]) & (dist_streamer.value<radiuses[i+1]))]
    # mask = np.where((distance_map>np.amin(xzrange.value)) & (distance_map<np.amax(xzrange.value)))
    # aux_max = np.amax(xzrange.value)
    # mask = np.where((distance_map>aux_min) & (distance_map<=aux_max))
    # radius3Dmap[mask] = binradii[i]
    # aux_min = aux_max
    NH2tot = np.nansum(NH2map[mask])
    NH2totkink = np.nansum(NH2mapkink[mask])
    masses[i] = M_hydrogen2(NH2tot, mu_H2, distance, deltara, deltadec)
    masseskink[i] = M_hydrogen2(NH2totkink, mu_H2, distance, deltara, deltadec)
    times[i] = t_freefall(binradii[i], Mstar)
    m_acclist[i] = masses[i] / times[i,1].value
    m_acclistkink[i] = masseskink[i] / times[i,1].value
# indexnans = np.where(np.isnan(unumpy.nominal_values(mom0kink)))
# radius3Dmap[indexnans] = np.nan
# if not os.path.exists(radius3Dmapname):
#     radius3Dheader = NC18Oheader.copy()
#     radius3Dheader['bunit'] = 'au'
#     hdu = fits.PrimaryHDU(data=radius3Dmap, header=radius3Dheader)
#     hdu.writeto(radius3Dmapname)

mass_streamer_table['3D distance (au)'] = binradii
mass_streamer_table['Mass wo kink (Msun)'] = unumpy.nominal_values(masses)
mass_streamer_table['u Mass wo kink (Msun)'] = unumpy.std_devs(masses)
mass_streamer_table['Mass w kink (Msun)'] = unumpy.nominal_values(masseskink)
mass_streamer_table['u Mass w kink (Msun)'] = unumpy.std_devs(masseskink)
mass_streamer_table['t_ff (yr, M_env = 0.39 Msun)'] = times[:,1]
mass_streamer_table['Mdot wo kink (Msun yr-1)'] = unumpy.nominal_values(m_acclist)
mass_streamer_table['u Mdot wo kink (Msun yr-1)'] = unumpy.std_devs(m_acclist)
mass_streamer_table['Mdot w kink (Msun yr-1)'] = unumpy.nominal_values(m_acclistkink)
mass_streamer_table['u Mdot w kink (Msun yr-1)'] = unumpy.std_devs(m_acclistkink)
mass_streamer_table.to_csv(tablefilemacc)

if not os.path.exists(radiusmapname):
    radiusheader = NC18Oheader.copy()
    radiusheader['bunit'] = 'au'
    hdu = fits.PrimaryHDU(data=distance_map, header=radiusheader)
    hdu.writeto(radiusmapname)
