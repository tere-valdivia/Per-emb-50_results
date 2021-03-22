import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.constants import G
from astropy.wcs import WCS
from astropy.io import fits

def v_kepler(mass, radius):
    vel = np.sqrt(G * mass / radius)
    return vel


pvfile = '../SO_55_44/CDconfig/pvex_Per-emb-50_CD_l009l048_uvsub_SO_multi_pbcor_pvline_center.fits'
v_lsr = 7.48*u.km/u.s  # +- 0.14 km/s according to out C18O data
arcsectoau = 293 # * u.au / u.arcsec
pvdata = fits.getdata(pvfile)
pvheader = fits.getheader(pvfile)

# The position velocity file is designed for the middle of the offset array to
# be the position of the protostar

delta0 = pvheader['CRVAL1']
delta_delta = pvheader['CDELT1']
delta_pix0 = pvheader['CRPIX1']
delta_npix = pvheader['NAXIS1']
vel0 = pvheader['CRVAL2']
delta_vel = pvheader['CDELT2']
vel_pix0 = pvheader['CRPIX2']
vel_npix = pvheader['NAXIS2']

delta_array = np.array([delta0 + delta_delta*(i-delta_pix0) for i in range(delta_npix)]) * u.deg
vel_array = np.array([vel0 + delta_vel * (i - vel_pix0) for i in range(vel_npix)]) * u.m/u.s

# transformation to general coordinates
vel_array = vel_array.to(u.km/u.s)
mid_delta = delta_array[int(len(delta_array)/2+1)]
offset_array = (delta_array - mid_delta).to(u.arcsec)
distance_array = offset_array.value * arcsectoau * u.au

offset, vel = np.meshgrid(distance_array,vel_array)
rms = 0.01
contourlevels = np.array([5,15, 25]) * rms

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
pcolor = ax.pcolor(offset.value, vel.value, pvdata, shading='auto', vmin=0)
contours = ax.contour(offset.value, vel.value, pvdata, contourlevels, colors='w')
fig.colorbar(pcolor, ax=ax)


#Now we plot a kepler rotation over it
mstar = [0.4, 0.8, 1.2] * u.Msun
colors = ['red','orange','brown']
radius = np.linspace(10,1500,50) * u.au
radius_neg = np.linspace(-10,-1500,50) * u.au
for mass,color in zip(mstar,colors):
    velocity = v_kepler(mass, radius).to(u.km/u.s) + v_lsr
    velocity_neg = -1*v_kepler(mass, radius).to(u.km/u.s) + v_lsr
    ax.plot(radius, velocity, ls='--', color=color, label=r'$M_{\star}='+str(mass.value)+'M_{\odot}$')
    ax.plot(radius_neg, velocity_neg, ls='--',color=color)

ax.set_ylim([0,14])
ax.set_ylabel(r'$v_{LSR}$ (km s$^{-1}$)')
ax.set_xlim([-1200,1200])
ax.set_xlabel('Offset distance (AU)')
ax.legend()

# fig.savefig('PV_diagram_with_Kepler_rot.pdf',bbox_inches='tight')
