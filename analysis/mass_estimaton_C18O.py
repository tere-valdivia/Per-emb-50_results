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
import aplpy
import copy

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
    Mass = N * (mu * m_p) * (D**2) * np.abs(deltara * deltadec)
    return Mass.to(u.Msun)

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

    TdVin must be in K km/s
    '''
    nu = 219560.3541 * u.MHz
    constant = 3 * h / (8*np.pi**3 * (1.1079e-19 *u.esu *u.cm)**2 *2/5)
    Eu = 15.81 * u.K
    NC18O = constant * Qrot(B0, Tex)/5 * np.exp(Eu / Tex) / \
        (np.exp(10.54*u.K/Tex)-1) * 1/(J_nu(nu, Tex) - J_nu(nu, 2.73*u.K)) * TdV/f
    return NC18O.to(u.cm**(-2))

# constant = 3 * h / (8*np.pi**3 * (1.1079e-19 *u.esu *u.cm)**2 *2/5)
# constant.decompose().to(u.s/u.km/u.cm**2)

'''
Inputs
'''
# filenameH2CO = '../H2CO/CDconfigsmall/Per-emb-50_CD_l021l060_uvsub_H2CO_multi_small_fitcube_fitted'
filenameC18O = '../' + C18O_2_1 + '_pbcor_reprojectH2COs_mom0_l'
tablefile = 'M_H2_Tex_fixed_mom0_pbcor.csv'
tablefilekink = 'M_H2_Tex_fixed_mom0_pbcor_kink.csv'
# snratio = 1
# rms = 13.94 * u.mJy/u.beam
# rms = 0.347 * u.K
NC18Ofilename = 'N_C18O_constantTex_{0}K_mom0_pbcor.fits'
NC18Ofilenamekink = 'N_C18O_constantTex_{0}K_mom0_pbcor_kink.fits'
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
NC18Oheader['bunit'] = 'cm-2'

if os.path.exists(tablefile):
    results_mass = pd.read_csv(tablefile)
    results_mass_kink = pd.read_csv(tablefilekink)
else:
    results_mass = pd.DataFrame(index=Texlist.value)

    for Tex in Texlist:
        # Do a N(C18O) map
        NC18O = N_C18O_21(mom0, B0, Tex) # the mom0 must have K km/s units
        #We save the column density obtained in a fits file
        if not os.path.exists('column_dens_maps/'+NC18Ofilename.format(Tex.value)):
            newfitshdu = fits.PrimaryHDU(data=NC18O.value, header=NC18Oheader)
            newfitshdu.writeto('column_dens_maps/'+NC18Ofilename.format(Tex.value))
        # We plot the column density
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection=wcsmom)
        im = ax.imshow(NC18O.value)
        fig.colorbar(im,ax=ax,label=r'N(C$^{18}$O) (cm$^{-2}$)')
        ax.set_xlabel('RA (J2000)')
        ax.set_ylabel('DEC (J2000)')
        if not os.path.exists('column_dens_maps/'+NC18Oplotname.format(Tex.value)):
            fig.savefig('column_dens_maps/'+NC18Oplotname.format(Tex.value))

        # Calculate statistics for future reference
        results_mass.loc[Tex.value, 'Mean NC18O (cm-2)'] = np.nanmean(NC18O.value)
        results_mass.loc[Tex.value, 'Standard deviation NC18O (cm-2)'] = np.nanstd(NC18O.value)
        results_mass.loc[Tex.value, 'Median NC18O (cm-2)'] = np.nanmedian(NC18O.value)
        results_mass.loc[Tex.value, 'Min NC18O (cm-2)'] = np.nanmin(NC18O.value)
        results_mass.loc[Tex.value, 'Max NC18O (cm-2)'] = np.nanmax(NC18O.value)
        # Now, we calculate the column density of H2
        results_mass.loc[Tex.value, 'Sum NC18O (cm-2 Npx)'] = NC18O.nansum().value
        NH2 = NC18O * X_C18O
        NH2tot = np.nansum(NH2)
        results_mass.loc[Tex.value, 'Sum NH2 (cm-2 Npx)'] = NH2tot.value
        MH2 = NH2tot * (mu_H2 * m_p) * (distance**2) * np.abs(deltara * deltadec)
        results_mass.loc[Tex.value, 'M (kg)'] = (MH2.to(u.kg)).value
        results_mass.loc[Tex.value, 'M (M_sun)'] = (MH2.to(u.Msun)).value
        print(Tex, MH2, MH2.to(u.Msun))

    results_mass.to_csv(tablefile)

NC18Omap = fits.getdata('column_dens_maps/'+NC18Ofilename.format(10.0))
NC18Omapkink = fits.getdata('column_dens_maps/'+NC18Ofilenamekink.format(10.0))
wcsmap = WCS(fits.getheader('column_dens_maps/'+NC18Ofilename.format(10.0))).celestial

NH2map = NC18Omap * X_C18O * (u.cm**-2)
NH2mapkink = NC18Omapkink * X_C18O * (u.cm**-2)
leny, lenx = np.shape(NH2map)

# This is to have an idea of the total mass
M_s = 1.71*u.Msun
M_env = np.array([0.18,0.39])*u.Msun
M_disk = 0.58*u.Msun
Mstar = (M_s+M_env+M_disk)
t_ff = t_freefall(3300*u.AU, Mstar)
leny, lenx = np.shape(NC18Omap)

M_acc = results_mass['M (M_sun)'].values * u.Msun
M_dot = [M_acc / t_ff[0], M_acc / t_ff[1]]


# Now, we separate the streamer in bins

xx, yy = np.meshgrid(range(lenx), range(leny))

distance_map = distance_physical(ra_Per50.value,dec_Per50.value, xx, yy, NC18Oheader)
binsize = 200 # au

radiuses = np.arange(0,3100, binsize)
binradii = np.arange(100,3100,binsize) # u.AU
masses = np.zeros(len(binradii)) * u.Msun
masseskink = np.zeros(len(binradii)) * u.Msun
times = np.zeros((len(binradii),2)) * u.yr
m_acclist = np.zeros((len(binradii),2)) * u.Msun / u.yr
m_acclistkink = np.zeros((len(binradii),2)) * u.Msun / u.yr

for i in range(len(radiuses)-1):
    mask = np.where((distance_map>radiuses[i]) & (distance_map<radiuses[i+1]))
    NH2tot = np.nansum(NH2map[mask])
    NH2totkink = np.nansum(NH2mapkink[mask])
    masses[i] = M_hydrogen2(NH2tot, mu_H2, distance, deltara, deltadec)
    masseskink[i] = M_hydrogen2(NH2totkink, mu_H2, distance, deltara, deltadec)
    times[i] = t_freefall(binradii[i]*u.AU, Mstar)
    m_acclist[i] = [masses[i] / times[i,0], masses[i] / times[i,1]]
    m_acclistkink[i] = [masseskink[i] / times[i,0], masseskink[i] / times[i,1]]


fig = plt.figure(figsize=(6,6))
ax3 = fig.add_subplot(313)
ax3.scatter(binradii, m_acclist[:,1].value, s=50, facecolors='none', edgecolors='r', label='No kink')
ax3.scatter(binradii, m_acclistkink[:,1].value, s=50, facecolors='r', edgecolors='r',label='kink')
ax3.set_yscale('log')
ax3.set_xlabel('Distance from Protostar (au)')
ax3.set_ylabel(r'$\dot{M}$ (M$_{\odot}$ yr$^{-1}$)')
ax3.annotate('bin size = {} au'.format(binsize), (0.6,0.8), xycoords='axes fraction', size=14)

ax = fig.add_subplot(311, sharex=ax3)
ax.scatter(binradii, masses.value*1e3, s=50, facecolors='none', edgecolors='r', label='No kink')
ax.scatter(binradii, masseskink.value*1e3, s=50, facecolors='r', edgecolors='r',label='kink')
ax.set_ylabel(r'Mass in bin ($\times 10^3$ M$_{\odot}$)')
ax.legend(loc=4)

ax2 = fig.add_subplot(312, sharex=ax3)
ax2.scatter(binradii, times[:,1].value, s=50, facecolors='b', edgecolors='b', label=r'$M_{*}=$'+str(Mstar[1]))
ax2.set_ylabel(r'Free-fall timescale (yr$^{-1}$)')
ax2.set_yscale('log')
ax2.legend()
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax.get_xticklabels(), visible=False)

fig.savefig('column_dens_maps/plot_mass_accretion_radius.pdf', dpi=300, bbox_inches='tight')

ra0_pix, dec0_pix = wcsmap.all_world2pix(ra_Per50.value, dec_Per50.value, 0)

def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s} %"

fig2 = plt.figure(figsize=(4,4))
ax = fig2.add_subplot(111, projection=wcsmap)
cmap = copy.copy(plt.cm.viridis)
cmap.set_bad(np.array((1,1,1))*0.85)
im = ax.imshow(fits.getdata('column_dens_maps/'+NC18Ofilename.format(10.0)), cmap=cmap)
plt.colorbar(im, ax=ax, label=r'N(C$^{18}$O) (cm$^{-2}$)')
cs = ax.contour(distance_map, levels=radiuses, colors='k', linewidths=1, linestyles='dashed')
ax.clabel(cs, cs.levels[::3],inline=True,fmt=fmt,fontsize=15)

fig2.savefig('column_dens_maps/N_C18O_constantTex_{0}K_mom0_pbcor_distance.pdf'.format(10.0))

fig3 = plt.figure(figsize=(4,4))
ax = fig3.add_subplot(111, projection=wcsmap)
im = ax.imshow(fits.getdata('column_dens_maps/'+NC18Ofilenamekink.format(10.0)), cmap=cmap)
plt.colorbar(im, ax=ax, label=r'N(C$^{18}$O) (cm$^{-2}$)')
cs = ax.contour(distance_map, levels=radiuses, colors='k', linewidths=1, linestyles='dashed')
ax.clabel(cs, cs.levels[::3],inline=True,fmt=fmt,fontsize=15)
fig3.savefig('column_dens_maps/N_C18O_constantTex_{0}K_mom0_pbcor_kink_distance.pdf'.format(10.0))

# gc = aplpy.FITSFigure('column_dens_maps/'+NC18Ofilenamekink.format(10.0), figure=fig3)
# gc.show_colorscale()
# gc.add_colorbar()
# distancehdu = fits.PrimaryHDU(data=distance_map, header=NC18Oheader)
# gc.show_contour(distancehdu,levels=radiuses,colors='k', linewidths=0.5, linestyles='dashed')
# plt.imshow(distance_map, origin='lower')

# for x in range(lenx):
#     for y in range(leny):
#         ra, dec = wcsmom
