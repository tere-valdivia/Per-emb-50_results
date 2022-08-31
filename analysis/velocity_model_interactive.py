"""
Author: Teresa Valdivia-Mena
Last revised August 31, 2022

This code an interactive interface to look at resulting streamlines, given the
one image containing the peak values and another image containing the central
velocities of the molecular line of interest. There are two parameters of the
model that are fixed: the inclination and position angle of the streamline. The
recommended values are the inclination and position angle of the protostellar
disk, if known.

This code makes use of the velocity_tools package by J. E. Pineda.
In the future, the plan is to include this interactive interface in the
velocity_tools package.

Current state: H2CO
"""

import sys
sys.path.append('../')
import pyregion
from astropy.coordinates import SkyCoord, FK5
from NOEMAsetup import *
from astropy.io import fits
from astropy.wcs import WCS
import velocity_tools.stream_lines as SL
import astropy.units as u
from scipy import stats
import pickle
from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt
import numpy as np

# Main parameters to generate a streamline
# For C18O, we try first with the minimum mass possible
M_s = 1.71*u.Msun # was 2.9
M_env = 0.18*u.Msun # was 2.2, 0.18 within 3500 AU
M_disk = 0.58*u.Msun
Mstar = (M_s+M_env+M_disk)
# Mstar = M_s
# Disk inclination system (i=0 is edge on)
inc = -(90-67) * u.deg
PA_ang = (170+90)*u.deg
# Proposed C18O inclination system (-170 deg)
# inc = -(43) * u.deg
# PA_ang = (20+90)*u.deg
regionsample = 'data/region_streamer_l_kink.reg'
# regionsample = 'data/region_streamer_C18O_final.reg'
imagename = '../'+H2CO_303_202_TdV+'.fits' #  C18O_2_1_TdV
vcname = '../'+H2CO_303_202_fit_Vc+'.fits' #  C18O_2_1_fit_Vc
# Fixed parameter
v_lsr = 7.48*u.km/u.s  # +- 0.14 km/s according to out C18O data

Per50_c = SkyCoord(ra_Per50, dec_Per50, frame='fk5')
Per50_ref = Per50_c.skyoffset_frame()

# Define the figure where the widgets will be
fig = plt.figure(figsize=(10, 7))
plt.subplots_adjust(left=0.1, bottom=0.45)

# Open the image plane
# hdu = fits.open('../'+H2CO_303_202_TdV_s+'.fits')
hdu = fits.open(imagename)

header = hdu[0].header
wcs = WCS(header)

# Plot the image plane in one of the axes
ax = fig.add_subplot(121, projection=wcs)
ax.imshow(hdu[0].data, vmin=0, vmax=4, origin='lower', cmap='Greys')
ax.set_autoscale_on(False)
ax.plot(ra_Per50, dec_Per50, transform=ax.get_transform('fk5'), marker='*',
        color='red')
hdu.close()
ax.set_xlabel('Right Ascension (J2000)')
ax.set_ylabel('Declination (J2000)')
# In case we want to zoom in
ra = 52.2813698
dec = 31.3648759
radiusplot = 12. / 3600.
zoomlims = wcs.all_world2pix([ra-radiusplot, ra+radiusplot],
                             [dec-radiusplot, dec+radiusplot], 0)
ax.set_xlim(zoomlims[0][1], zoomlims[0][0])
ax.set_ylim(zoomlims[1][0], zoomlims[1][1])

regstreamer = pyregion.open('../'+regionsample)
r2 = regstreamer.as_imagecoord(header)
patch_list, artist_list = r2.get_mpl_patches_texts()
for p in patch_list:
    ax.add_patch(p)
for a in artist_list:
    ax.add_artist(a)

# We add the axes to the image plane
x_b = np.array([1, 0, 0])*1e3/dist_Per50
y_b = np.array([0, 0, 0])*1e3/dist_Per50
z_b = np.array([0, 0, 1])*1e3/dist_Per50
nx_b, ny_b, nz_b = SL.rotate_xyz(x_b, y_b, z_b, inc=inc, pa=PA_ang)
# original axes
my_axis = SkyCoord(-x_b*u.arcsec, z_b*u.arcsec,
                   frame=Per50_ref).transform_to(FK5)
ax.plot(my_axis.ra, my_axis.dec, transform=ax.get_transform('fk5'),
        color='k')
# new axes
my_axis_new = SkyCoord(-nx_b*u.arcsec, nz_b*u.arcsec,
                             frame=Per50_ref).transform_to(FK5)
if ny_b[-1] > 0:
    new_ax_color = 'red'
else:
    new_ax_color = 'blue'
ax.plot(my_axis_new.ra[1:], my_axis_new.dec[1:], transform=ax.get_transform('fk5'),
        color=new_ax_color)
ax.plot(my_axis_new.ra[0:2], my_axis_new.dec[0:2], transform=ax.get_transform('fk5'),
        color='red')


# Plot the velocity plane in the other axis
ax3 = fig.add_subplot(122)
ax3.set_xlabel('Projected distance (au)')
ax3.set_ylabel(r"V$_{lsr}$ (km s$^{-1}$)")

# We obtain the kernel probability distribution for r and v

# Obtain the offset radius from Per-emb-50 and the v_lsr for each
# pixel in the streamer region
r_proj, v_los = per_emb_50_get_vc_r(vcname, '../'+regionsample)
# create the grid for the kernel distribution
# x is projected distance
xmin = 0
xmax = 5000
# y is velocity lsr
ymin = 6.5
ymax = 8.5

xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
# we select only those who are not nan
gd_vlos = np.isfinite(r_proj*v_los)
values = np.vstack([r_proj[gd_vlos].value, v_los[gd_vlos].value])
# we calculate the kernel distribution
kernel = stats.gaussian_kde(values)
zz = np.reshape(kernel(positions).T, xx.shape)
zz /= zz.max()  # normalization of probability
# We plot in the corresponding axis
ax3.contourf(xx, yy, zz, cmap='Greys', levels=np.arange(0.1, 1.2, 0.1), vmin=0., vmax=1.1)
ax3.axhline(v_lsr.value, color='k')
ax3.set_ylim([ymin,ymax])
ax3.set_xlim([xmin,xmax])

# We calculate the streamlines for several parameters


def get_streamer(mass, r0, theta0, phi0, omega0, v_r0, inc, PA):
    (x1, y1, z1), (vx1, vy1, vz1) = SL.xyz_stream(
        mass=mass, r0=r0, theta0=theta0, phi0=phi0,
        omega=omega0, v_r0=v_r0, inc=inc, pa=PA) #, deltar=10*u.au)
    # we obtain the distance of each point in the sky
    d_sky_au = np.sqrt(x1**2 + z1**2)
    # Stream line into arcsec
    dra_stream = -x1.value / dist_Per50
    ddec_stream = z1.value / dist_Per50
    fil = SkyCoord(dra_stream*u.arcsec, ddec_stream*u.arcsec,
                   frame=Per50_ref).transform_to(FK5)
    velocity = v_lsr + vy1
    return fil, d_sky_au, velocity

# Initial parameters
# H2CO
theta0 = 61.5 * u.deg  # rotate clockwise
phi0 = 28 * u.deg  # rotate the plane
v_r0 = 1.25 * u.km/u.s
omega0 = 4.53e-13 / u.s
r0 = 3330 * u.au
# Test for C18O
# theta0 = 89.5 * u.deg  # rotate clockwise
# r0 = 3670*u.au
# phi0 = 162.0 * u.deg
# v_r0 = 0.05 * u.km/u.s
# omega0 = 8.72e-13 / u.s

# Parameter steps
delta_theta0 = 0.5
delta_phi0 = 0.5
delta_r0 = 10
delta_omega0 = 0.5e-15
delta_v_r0 = 0.05

# We calculate the initial streamer
fil0, dsky0, velo0 = get_streamer(Mstar, r0, theta0, phi0, omega0, v_r0, inc, PA_ang)
r_c = SL.r_cent(Mstar,omega0,r0)
annotation = ax3.annotate(r'$r_c = {}$'.format(np.round(r_c,0)), (0.6, 0.1), xycoords='axes fraction', size=12)

line_image, = ax.plot(fil0.ra, fil0.dec, transform=ax.get_transform('fk5'),
                      ls='-', lw=2)
line_vel, = ax3.plot(dsky0, velo0)
#
# line_old_image, = ax.plot(fil0.ra, fil0.dec, transform=ax.get_transform('fk5'),
#                       ls='--', lw=2)
# line_old_vel, = ax3.plot(dsky0, velo0,ls='--')

# We create the sliders
axcolor = 'paleturquoise'
axtheta0 = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor=axcolor)
stheta0 = Slider(axtheta0, r'$\theta_0$', 0, 180., valinit=theta0.value, valstep=delta_theta0)
axphi0 = plt.axes([0.2, 0.15, 0.6, 0.03], facecolor=axcolor)
sphi0 = Slider(axphi0, r'$\phi_0$', -90, 270, valinit=phi0.value, valstep=delta_phi0)
axr0 = plt.axes([0.2, 0.2, 0.6, 0.03], facecolor=axcolor)
sr0 = Slider(axr0, r'$r_0$', 1500, 5000, valinit=r0.value, valstep=delta_r0)
axomega0 = plt.axes([0.2, 0.25, 0.6, 0.03], facecolor=axcolor)
somega0 = Slider(axomega0, r'$\Omega_0$', 1.e-14, 15e-13, valinit=omega0.value, valstep=delta_omega0)
axv0 = plt.axes([0.2, 0.3, 0.6, 0.03], facecolor=axcolor)
sv0 = Slider(axv0, r'$v_{r,0}$', 0, 5, valinit=v_r0.value, valstep=delta_v_r0)

# In the update function, add the parameters per slider

def update(val):
    theta = stheta0.val * u.deg
    phi = sphi0.val * u.deg
    # r_c = src0.val * u.au
    omega = somega0.val / u.s
    rnew = sr0.val * u.au
    v_r = sv0.val * u.km / u.s
    fil, dsky, velo = get_streamer(Mstar, rnew, theta, phi, omega, v_r, inc, PA_ang)
    line_image.set_xdata(fil.ra)
    line_image.set_ydata(fil.dec)
    line_vel.set_xdata(dsky)
    line_vel.set_ydata(velo)
    r_cnew = SL.r_cent(Mstar, omega, rnew)
    annotation.set_text(r'$r_c = {}$'.format(np.round(r_cnew,0)))
    fig.canvas.draw_idle()


updateax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(updateax, 'Update', color=axcolor, hovercolor='0.975')
button.on_clicked(update)

plt.show()
