import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.constants import G
from astropy.wcs import WCS
from astropy.io import fits
from astropy.visualization import simple_norm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

def v_kepler(mass, radius):
    vel = np.sqrt(G * mass / radius)
    return vel


pvfile = '../SO_55_44/CDconfig/pvex_Per-emb-50_CD_l009l048_uvsub_SO_multi_pbcor_pvline_center_Per50_1arcsec_170PA.fits'
# pvfile = '../C18O/CDconfig/JEP/pvex_JEP_mask_multi_Per-emb-50_CD_l025l064_uvsub_C18O_pvline_center_Per50_1arcsec_170PA_12arcsec.fits'
v_lsr = 7.48*u.km/u.s  # +- 0.14 km/s according to out C18O data
arcsectoau = 293  # * u.au / u.arcsec
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
mid_delta = delta_array[int(len(delta_array)/2+2-1)]
offset_array = (delta_array - mid_delta).to(u.arcsec)
distance_array = offset_array.value * arcsectoau * u.au

offset, vel = np.meshgrid(distance_array, vel_array)
rms = 0.01
contourlevels = np.array([5,15,25]) * rms
vmin = 0
vmax = 0.4

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
norm = simple_norm(pvdata, 'linear', min_cut=vmin,max_cut=vmax)
pcolor = ax.pcolor(offset.value, vel.value, pvdata, shading='auto', norm=norm, cmap='Blues')
contours = ax.contour(offset.value, vel.value, pvdata, contourlevels, colors='k', linewidths=0.5)
fig.colorbar(pcolor, ax=ax, label=r'Intensity (Jy beam$^{-1}$)')


# Now we plot a kepler rotation over it
# mstar = [0.4] * u.Msun
mstar = [0.5, 0.7, 1.5, 1.9] * u.Msun
inclination = 67 # 0 is face on
colors = ['red', 'orange', 'red', 'orange']
linestyles = ['-', '-', '--', '--']
radius = np.linspace(1, 1600, 1000) * u.au
radius_neg = np.linspace(-1, -1600, 1000) * u.au
for mass, color, ls in zip(mstar, colors,  linestyles):
    # velocity = v_kepler(mass, radius).to(u.km/u.s) + v_lsr
    velocity = v_kepler(mass, radius).to(u.km/u.s) * np.sin(inclination*np.pi/180)
    velocity_pos = velocity + v_lsr
    velocity_neg = -1*velocity + v_lsr
    ax.plot(radius, velocity_pos, ls=ls, color=color,
            label=r'$M_{\star}='+str(mass.value)+r'M_{\odot}$')
    ax.plot(radius_neg, velocity_neg, ls=ls, color=color)
ax.axhline(v_lsr.value,color='k', linestyle=':', linewidth=3)
ax.set_ylim([0, 14])
ax.set_ylabel(r'$v_{LSR}$ (km s$^{-1}$)')
ax.set_xlim([-1200, 1200])
ax.set_xlabel('Offset distance (AU)')
ax.legend(fontsize=8, loc=4)
ax.annotate(r'rms = 0.01 Jy beam$^{-1}$', (0.05, 0.05), xycoords='axes fraction', color='k', size=8)
ax.annotate(r'i = 67$^{\circ}$', (0.05, 0.01), xycoords='axes fraction', color='k', size=8)

bar = AnchoredSizeBar(ax.transData, 300, '300 AU', 2,pad=0.1, borderpad=0.5, sep=5,  frameon=False, color='k', size_vertical=0.08)
ax.add_artist(bar)
fig.savefig('PV_diagram_SO_170_with_Kepler_rot_length1600AU_Fiorellino_21.pdf', dpi=300, bbox_inches='tight')
