import sys
sys.path.append('../')

from scipy import stats
import numpy as np
import astropy.units as u
import velocity_tools.stream_lines as SL
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits
from NOEMAsetup import *
from astropy.coordinates import SkyCoord, FK5
import pyregion


# This import registers the 3D projection, but is otherwise unused.

'''
This code is to test different combinations of parameters to get the best
ones for the streamer in Per-emb-50
Aim for a r_c about 100 AU

in this code, i=0 is an edge on disk
'''
# Main parameters to generate a streamline
# Mstar = 0.58*u.Msun
Mstar = (2.9+2.2+0.58)*u.Msun # mass of the star and envelope and disk
# inc = -(67-180)*u.deg
inc = (360-(90-67))*u.deg # should be almost edge on
PA_ang = -(170-90)*u.deg
regionsample = 'data/region_streamer_test.reg'

# Fixed parameter
v_lsr = 7.48*u.km/u.s #+- 0.14 km/s according to out C18O data

Per50_c = SkyCoord(ra_Per50, dec_Per50, frame='fk5')
Per50_ref = Per50_c.skyoffset_frame()

# figure 1: 3d plot to see the 3d structure of the streamline
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Figure 2: plot to observe the streamer in the image plan
hdu = fits.open('../'+H2CO_303_202_TdV_s+'.fits')
header = hdu[0].header
freq_H2CO_303_202 = header['RESTFREQ'] * u.Hz
wcs = WCS(header)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection=wcs)
ax2.imshow(hdu[0].data, vmin=0, vmax=4, origin='lower', cmap='Greys')
ax2.set_autoscale_on(False)
ax2.plot(ra_Per50, dec_Per50, transform=ax2.get_transform('fk5'), marker='*',
         color='red')
# ax2.set_title(r'$M = {}$'.format(Mstar))
hdu.close()
ax2.set_xlabel('Right Ascension (J2000)')
ax2.set_ylabel('Declination (J2000)')
regstreamer = pyregion.open('../'+regionsample)
r2 = regstreamer.as_imagecoord(header)
patch_list, artist_list = r2.get_mpl_patches_texts()
for p in patch_list:
    ax2.add_patch(p)
for a in artist_list:
    ax2.add_artist(a)
# In case we want to zoom in
ra = 52.2813698
dec = 31.3648759
radiusplot = 10. / 3600.
zoomlims = wcs.all_world2pix([ra-radiusplot, ra+radiusplot],
                             [dec-radiusplot, dec+radiusplot], 0)
ax2.set_xlim(zoomlims[0][1], zoomlims[0][0])
ax2.set_ylim(zoomlims[1][0], zoomlims[1][1])

# Figure 3: plot to observe the streamer in velocity
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
# axes labels
ax3.set_xlabel('Projected distance (au)')
ax3.set_ylabel(r"V$_{lsr}$ (km s$^{-1}$)")

# First we add the axes to the image plane to see the dimensions we are
# working with
x_b = np.array([1, 0, 0])*1e3/dist_Per50
y_b = np.array([0, 0, 0])*1e3/dist_Per50
z_b = np.array([0, 0, 1])*1e3/dist_Per50
nx_b, ny_b, nz_b = SL.rotate_xyz(x_b, y_b, z_b, inc=inc, pa=PA_ang)

# original axes
my_axis = SkyCoord(-x_b*u.arcsec, z_b*u.arcsec,
                   frame=Per50_ref).transform_to(FK5)
ax2.plot(my_axis.ra, my_axis.dec, transform=ax2.get_transform('fk5'),
         color='k')
# new axes
my_axis_new = SkyCoord(-nx_b*u.arcsec, nz_b*u.arcsec,
                             frame=Per50_ref).transform_to(FK5)

if ny_b[-1] > 0:
    new_ax_color = 'red'
else:
    new_ax_color = 'blue'

# ax2.plot(my_axis_new.ra, my_axis_new.dec, transform=ax2.get_transform('fk5'),
#          color=new_ax_color)

ax2.plot(my_axis_new.ra[1:], my_axis_new.dec[1:], transform=ax2.get_transform('fk5'),
                     color=new_ax_color)
ax2.plot(my_axis_new.ra[0:2], my_axis_new.dec[0:2], transform=ax2.get_transform('fk5'),
         color='red')

# We obtain the kernel probability distribution for r and v

# Obtain the offset radius from Per-emb-50 and the v_lsr for each
# pixel in the streamer region
r_proj, v_los = per_emb_50_get_vc_r('../'+H2CO_303_202_fit_Vc+'.fits',
                                    '../'+regionsample)
# create the grid for the kernel distribution
#x is projected distance
xmin = 0
xmax = 4000
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

# We calculate the streamlines for several parameters


# Constant parameters for testing
theta0 = 89.9*u.deg  # rotate clockwise
r0 = 3800.*u.au
phi0 = 20.*u.deg  # rotate the plane
v_r0 = 1.*u.km/u.s
omega0 = 5e-13/u.s

# Arrays
# 89.9, 95, 100, 105
# 70, 75, 80, 85
thetalist = np.array([70, 75, 80, 85]) * u.deg
rlist = np.array([1500, 1600, 1700, 1800, 1900])* u.au
philist = np.array([0, 10, 20, 30, 40])* u.deg
# rlist =np.array([1600, 1800, 2000]) * u.au
v_rlist = np.array([4, 5, 6]) * u.km/u.s
omegalist = np.array([1,4,7,9])* 1.e-13 / u.s

# Make a label for each streamline set of parameters
def stream_label(v_r=None, omega=None, theta=None, phi=None, r=None):
    my_label = ""
    if v_r is not None:
        my_label = r"{0} $V_r=${1}".format(my_label, v_r)
    if omega is not None:
        my_label = r"{0} $\Omega=${1}".format(my_label, np.round(omega, 14))
    if theta is not None:
        my_label = r"{0} $\theta_0=${1}".format(my_label, theta)
    if phi is not None:
        my_label = r"{0} $\phi_0=${1}".format(my_label, phi)
    if r is not None:
        my_label = r"{0} $r_0=${1}".format(my_label, r)
    return my_label

#
# (x1, y1, z1), (vx1, vy1, vz1) = SL.xyz_stream(
#     mass=Mstar, r0=r0, theta0=theta0, phi0=phi0,
#     omega=omega0, v_r0=v_r0, inc=inc, pa=PA_ang, rmin=200*u.au)
# my_label = stream_label(omega=omega0, theta=theta0, phi=phi0, r=r0, v_r=v_r0)
# # we obtain the distance of each point in the sky
# d_sky_au = np.sqrt(x1**2 + z1**2)
# # Stream line into arcsec
# dra_stream = -x1.value / dist_Per50
# ddec_stream = z1.value / dist_Per50
# fil = SkyCoord(dra_stream*u.arcsec, ddec_stream*u.arcsec,
#                frame=Per50_ref).transform_to(FK5)
# # First we plot the 3d
# ax.plot(x1, y1, z1, marker='o', markersize=1, label=my_label)
# ax.plot(x1[0], y1[0], z1[0], marker='o', color='k')
# #Then we plot the streamer in the image plane
# ax2.plot(fil.ra, fil.dec, transform=ax2.get_transform('fk5'),
#          ls='-', lw=1, label=my_label)
# # Finally we plot the streamer in velocity
# ax3.plot(d_sky_au, v_lsr + vy1, label=my_label)


for phi0 in philist:
# for r0 in rlist:
# for v_r0 in v_rlist:
# for omega0 in omegalist:
# for theta0 in thetalist:
    #we obtain the streamline positions and velocities
    (x1, y1, z1), (vx1, vy1, vz1) = SL.xyz_stream(
        mass=Mstar, r0=r0, theta0=theta0, phi0=phi0,
        omega=omega0, v_r0=v_r0, inc=inc, pa=PA_ang, rmin=300*u.au)
    my_label = stream_label(omega=omega0, theta=theta0, phi=phi0, r=r0, v_r=v_r0)
    # we obtain the distance of each point in the sky
    d_sky_au = np.sqrt(x1**2 + z1**2)
    # Stream line into arcsec
    dra_stream = -x1.value / dist_Per50
    ddec_stream = z1.value / dist_Per50
    fil = SkyCoord(dra_stream*u.arcsec, ddec_stream*u.arcsec,
                   frame=Per50_ref).transform_to(FK5)
    # First we plot the 3d
    ax.plot(x1, y1, z1, marker='o', markersize=1, label=my_label)
    ax.plot(x1[0], y1[0], z1[0], marker='o', color='k')
    #Then we plot the streamer in the image plane
    ax2.plot(fil.ra, fil.dec, transform=ax2.get_transform('fk5'),
             ls='-', lw=1, label=my_label)
    # Finally we plot the streamer in velocity
    ax3.plot(d_sky_au, v_lsr + vy1, label=my_label)

# Plot legend at the end
# ax.legend()
ax2.legend(prop={'size': 8})
# ax3.legend()
plt.show()
