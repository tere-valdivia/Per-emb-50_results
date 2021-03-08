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
# Mstar = 0.58*u.Msun
Mstar = (2.9+2.2+0.58)*u.Msun  # mass of the star and envelope and disk
# inc = -(67-180)*u.deg
inc = (360-(90-67))*u.deg  # should be almost edge on
# inc = (360-(90-77))*u.deg # should be almost edge on
PA_ang = -(170-90)*u.deg
regionsample = 'data/region_streamer_l.reg'

# Fixed parameter
v_lsr = 7.48*u.km/u.s  # +- 0.14 km/s according to out C18O data

Per50_c = SkyCoord(ra_Per50, dec_Per50, frame='fk5')
Per50_ref = Per50_c.skyoffset_frame()

# Define the figure where the widgets will be
fig = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.25, bottom=0.25)

# Open the image plane
hdu = fits.open('../'+H2CO_303_202_TdV_s+'.fits')
header = hdu[0].header
freq_H2CO_303_202 = header['RESTFREQ'] * u.Hz
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
radiusplot = 10. / 3600.
zoomlims = wcs.all_world2pix([ra-radiusplot, ra+radiusplot],
                             [dec-radiusplot, dec+radiusplot], 0)
ax.set_xlim(zoomlims[0][1], zoomlims[0][0])
ax.set_ylim(zoomlims[1][0], zoomlims[1][1])

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
r_proj, v_los = per_emb_50_get_vc_r('../'+H2CO_303_202_fit_Vc+'.fits',
                                    '../'+regionsample)
# create the grid for the kernel distribution
# x is projected distance
xmin = 0
xmax = 3500
# y is velocity lsr
ymin = 6.
ymax = 8.

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
ax3.set_ylim([6,8])

# We calculate the streamlines for several parameters


def get_streamer(mass, r0, theta0, phi0, omega0, v_r0, inc, PA):
    (x1, y1, z1), (vx1, vy1, vz1) = SL.xyz_stream(
        mass=mass, r0=r0, theta0=theta0, phi0=phi0,
        omega=omega0, v_r0=v_r0, inc=inc, pa=PA, rmin=rmin)
    # we obtain the distance of each point in the sky
    d_sky_au = np.sqrt(x1**2 + z1**2)
    # Stream line into arcsec
    dra_stream = -x1.value / dist_Per50
    ddec_stream = z1.value / dist_Per50
    fil = SkyCoord(dra_stream*u.arcsec, ddec_stream*u.arcsec,
                   frame=Per50_ref).transform_to(FK5)
    velocity = v_lsr + vy1
    return fil, d_sky_au, velocity


def r0ideal(omega, mass, rcideal):
    r0i = (SL.G * Mstar * rcideal/(omega**2))**(1/4.)
    return r0i


# Initial parameters
theta0 = 80. * u.deg  # rotate clockwise
r_c0 = 300 * u.au
phi0 = 5. * u.deg  # rotate the plane
v_r0 = 0. * u.km/u.s
omega0 = 8e-13 / u.s
r0 = r0ideal(omega0, Mstar, r_c0).to(u.au)
# rmin = 300*u.au
rmin = r_c0
# Parameter steps
delta_theta0 = 1.

# We calculate the initial streamer
fil0, dsky0, velo0 = get_streamer(Mstar, r0, theta0, phi0, omega0, v_r0, inc, PA_ang)
line_image, = ax.plot(fil0.ra, fil0.dec, transform=ax.get_transform('fk5'),
                      ls='-', lw=1)
line_vel, = ax3.plot(dsky0, velo0)

# We create the sliders
axcolor = 'lightgoldenrodyellow'
axtheta0 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
stheta0 = Slider(axtheta0, r'$\theta_0$', 70., 89.9, valinit=theta0.value, valstep=delta_theta0)

def update(val):
    theta = stheta0.val* u.deg
    fil, dsky, velo = get_streamer(Mstar, r0, theta, phi0, omega0, v_r0, inc, PA_ang)
    line_image.set_xdata(fil.ra)
    line_image.set_ydata(fil.dec)
    line_vel.set_xdata(dsky)
    line_vel.set_ydata(velo)
    fig.canvas.draw_idle()


updateax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(updateax, 'Update', color=axcolor, hovercolor='0.975')
button.on_clicked(update)

plt.show()
