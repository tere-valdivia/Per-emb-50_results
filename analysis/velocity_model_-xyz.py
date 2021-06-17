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
import pickle

'''
This code is to test different combinations of parameters to get the best
ones for the streamer in Per-emb-50
Aim for a r_c about 300 AU

in this code, i=0 is an edge on disk
since correcting the angles, not ran again
'''

# Main parameters to generate a streamline
# The mass is currently in testing
M_s = 1.71*u.Msun # was 2.9
# M_env = 0.18*u.Msun # lower limit
M_env = 0.39*u.Msun # upper limit
M_disk = 0.58*u.Msun
# Mstar = (M_s+M_env+M_disk)
Mstar = 8.3 *u.Msun # test for
# Disk inclination system
# inc = -(90-67) * u.deg
# PA_ang = (170+90)*u.deg
# C18O proposed system
inc = -(90-67) * u.deg
PA_ang = (20+90)*u.deg
regionsample = 'data/region_streamer_C18O_test4.reg'
savekernel = False
savemodel = False

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

# Figure 2: plot to observe the streamer in the image plane
hdu = fits.open('../'+C18O_2_1_TdV+'.fits')
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

# For plotting the different regions together
# regionsamples = ['data/region_streamer_s.reg', 'data/region_streamer_m.reg', 'data/region_streamer_l.reg']
# for reg in regionsamples:
#     regstreamer = pyregion.open('../'+reg)
#     r2 = regstreamer.as_imagecoord(header)
#     patch_list, artist_list = r2.get_mpl_patches_texts()
#     for p in patch_list:
#         ax2.add_patch(p)
#     for a in artist_list:
#         ax2.add_artist(a)

regstreamer = pyregion.open('../'+regionsample)
r2 = regstreamer.as_imagecoord(header)
patch_list, artist_list = r2.get_mpl_patches_texts()
for p in patch_list:
    ax2.add_patch(p)
for a in artist_list:
    ax2.add_artist(a)

# In case we want to zoom in
# ra = 52.2813698
# dec = 31.3648759
# radiusplot = 10. / 3600.
# zoomlims = wcs.all_world2pix([ra-radiusplot, ra+radiusplot],
#                              [dec-radiusplot, dec+radiusplot], 0)
# ax2.set_xlim(zoomlims[0][1], zoomlims[0][0])
# ax2.set_ylim(zoomlims[1][0], zoomlims[1][1])

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


ax2.plot(my_axis_new.ra[1:], my_axis_new.dec[1:], transform=ax2.get_transform('fk5'),
                     color=new_ax_color)
ax2.plot(my_axis_new.ra[0:2], my_axis_new.dec[0:2], transform=ax2.get_transform('fk5'),
         color='red')

# We obtain the kernel probability distribution for r and v

# Obtain the offset radius from Per-emb-50 and the v_lsr for each
# pixel in the streamer region
r_proj, v_los = per_emb_50_get_vc_r('../'+C18O_2_1_fit_Vc+'.fits',
                                    '../'+regionsample)
# create the grid for the kernel distribution
#x is projected distance
xmin = 0
xmax = 5000
# y is velocity lsr
ymin = 6.
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
ax3.set_xlim([xmin, xmax])
# We calculate the streamlines for several parameters
# fig3.savefig('v_lsr_vs_rproj_C18O_reg_test4.pdf', dpi=300,bbox_inches='tight')

def r0ideal(omega, mass, rcideal):
    r0i = (SL.G * Mstar * rcideal/(omega**2))**(1/4.)
    return r0i

# Constant parameters for testing
theta0 = 89. * u.deg  # rotate clockwise
# r_c0 = 300 * u.au
# phi0 = 90. * u.deg  # rotate the plane
phi0 = 173.5 * u.deg  # rotate the plane
v_r0 = 0 * u.km/u.s
omega0 = 11e-13 / u.s
r0 = 3670*u.au
#r0 = r0ideal(omega0, Mstar, r_c0).to(u.au)
# print('The ideal r0 for '+str(omega0)+' is '+str(r0))


# Arrays
# 89.9, 95, 100, 105
# 70, 75, 80, 85
thetalist = np.array([70,80,90,100,110]) * u.deg
rlist = np.array([1500, 1600, 1700, 1800, 1900])* u.au
philist = np.array([-50])* u.deg
# rlist =np.array([1600, 1800, 2000]) * u.au
v_rlist = np.array([0,1,2]) * u.km/u.s
omegalist = np.array([4, 8, 12])* 1.e-13 / u.s
M_slist = np.array([1.5, 1.9, 2.9]) * u.Msun
M_elist = np.array([0.18, 2.2]) * u.Msun
M_totlist = np.array([(1.52 + 1.9)/2+0.18, 2.9+2.2]) * u.Msun
# colorlist = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Make a label for each streamline set of parameters
def stream_label(v_r=None, omega=None, theta=None, phi=None, r=None, M=None):
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
        my_label = r"{0} $r_0=${1}".format(my_label, np.round(r,0))
    if M is not None:
        my_label = r"{0} $M_e+M_s=${1}".format(my_label, M)
    return my_label

# Single parameters
(x1, y1, z1), (vx1, vy1, vz1) = SL.xyz_stream(
    mass=Mstar, r0=r0, theta0=theta0, phi0=phi0,
    omega=omega0, v_r0=v_r0, inc=inc, pa=PA_ang, rmin=100*u.au)
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

# # One array of parameters
# for phi0 in philist:
# # for r0 in rlist:
# # for v_r0 in v_rlist:
# # for omega0 in omegalist:
# # for theta0 in thetalist:
# # for mt in M_totlist:
#     # we obtain the streamline positions and velocities
#     # Only if omega0 or the mass varies
#     # Mstar = mt+M_disk
#     # r0 = r0ideal(omega0, Mstar, r_c0).to(u.au)
#     (x1, y1, z1), (vx1, vy1, vz1) = SL.xyz_stream(
#         mass=Mstar, r0=r0, theta0=theta0, phi0=phi0,
#         omega=omega0, v_r0=v_r0, inc=inc, pa=PA_ang)
#     my_label = stream_label(omega=omega0, theta=theta0, phi=phi0, r=r0)
#     # we obtain the distance of each point in the sky
#     d_sky_au = np.sqrt(x1**2 + z1**2)
#     # Stream line into arcsec
#     dra_stream = -x1.value / dist_Per50
#     ddec_stream = z1.value / dist_Per50
#     fil = SkyCoord(dra_stream*u.arcsec, ddec_stream*u.arcsec,
#                    frame=Per50_ref).transform_to(FK5)
#     # First we plot the 3d
#     print(vy1[0])
#     ax.plot(x1, y1, z1, marker='o', markersize=1, label=my_label)
#     # ax.plot(x1[0], y1[0], z1[0], marker='o', color='k')
#     #Then we plot the streamer in the image plane
#     ax2.plot(fil.ra, fil.dec, transform=ax2.get_transform('fk5'),
#              ls='-', lw=1, label=my_label)
#     # Finally we plot the streamer in velocity
#     ax3.plot(d_sky_au, v_lsr + vy1, label=my_label)
#
    #
    # (x1, y1, z1), (vx1, vy1, vz1) = SL.xyz_stream(
    #     mass=Mstar, r0=r0, theta0=theta0, phi0=phi0+5*u.deg,
    #     omega=omega0, v_r0=v_r0, inc=inc, pa=PA_ang, rmin=300*u.au)
    # my_label = stream_label(omega=omega0, theta=theta0, phi=phi0+5*u.deg, r=r0, v_r=v_r0)
    # # we obtain the distance of each point in the sky
    # d_sky_au = np.sqrt(x1**2 + z1**2)
    # # Stream line into arcsec
    # dra_stream = -x1.value / dist_Per50
    # ddec_stream = z1.value / dist_Per50
    # fil = SkyCoord(dra_stream*u.arcsec, ddec_stream*u.arcsec,
    #                frame=Per50_ref).transform_to(FK5)
    # # First we plot the 3d
    # print(vy1[0])
    # ax.plot(x1, y1, z1, linestyle='dashed', linewidth=1, label=my_label)
    # ax.plot(x1[0], y1[0], z1[0], marker='o', color='k')
    # #Then we plot the streamer in the image plane
    # ax2.plot(fil.ra, fil.dec, transform=ax2.get_transform('fk5'),
    #          ls='--', lw=1, label=my_label)
    # # Finally we plot the streamer in velocity
    # ax3.plot(d_sky_au, v_lsr + vy1, label=my_label, linestyle='dashed')

#
# # Arrays of selected parameters
# thetalist = np.array([73,75,78,81,84]) * u.deg
# philist = np.array([0,5,10,15,20]) * u.deg
# colorlist = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
#
# thetalist = thetalist[7:]
# rlist = rlist[7:]
# philist = philist[7:]
# colorlist = colorlist[:4]
#
#
# for theta0, phi0, color0 in zip(thetalist, philist, colorlist):
#     (x1, y1, z1), (vx1, vy1, vz1) = SL.xyz_stream(
#         mass=Mstar, r0=r0, theta0=theta0, phi0=phi0,
#         omega=omega0, v_r0=v_r0, inc=inc, pa=PA_ang, rmin=300*u.au)
#     my_label = stream_label(omega=omega0, theta=theta0, phi=phi0, r=r0, v_r=v_r0)
#     # we obtain the distance of each point in the sky
#     d_sky_au = np.sqrt(x1**2 + z1**2)
#     # Stream line into arcsec
#     dra_stream = -x1.value / dist_Per50
#     ddec_stream = z1.value / dist_Per50
#     fil = SkyCoord(dra_stream*u.arcsec, ddec_stream*u.arcsec,
#                    frame=Per50_ref).transform_to(FK5)
#     # First we plot the 3d
#     ax.plot(x1, y1, z1, linestyle='dashed', linewidth=1, label=my_label, color=color0)
#     ax.plot(x1[0], y1[0], z1[0], marker='o', color='k')
#     #Then we plot the streamer in the image plane
#     ax2.plot(fil.ra, fil.dec, transform=ax2.get_transform('fk5'),
#              ls='--', lw=1, label=my_label, color=color0)
#     # Finally we plot the streamer in velocity
#     ax3.plot(d_sky_au, v_lsr + vy1, label=my_label, color=color0, linestyle='dashed')
#
    # # Now we plot with a starting velocity
    # (x1, y1, z1), (vx1, vy1, vz1) = SL.xyz_stream(
    #     mass=Mstar, r0=r0, theta0=theta0, phi0=phi0,
    #     omega=omega0, v_r0=v_r0, inc=inc, pa=PA_ang, rmin=300*u.au)
    # my_label = stream_label(omega=omega0, theta=theta0, phi=phi0, r=r0, v_r=v_r0)
    # # we obtain the distance of each point in the sky
    # d_sky_au = np.sqrt(x1**2 + z1**2)
    # # Stream line into arcsec
    # dra_stream = -x1.value / dist_Per50
    # ddec_stream = z1.value / dist_Per50
    # fil = SkyCoord(dra_stream*u.arcsec, ddec_stream*u.arcsec,
    #                frame=Per50_ref).transform_to(FK5)
    # # First we plot the 3d
    # ax.plot(x1, y1, z1, marker='o', markersize=1, label=my_label, color=color0)
    # ax.plot(x1[0], y1[0], z1[0], marker='o', color='k')
    # #Then we plot the streamer in the image plane
    # ax2.plot(fil.ra, fil.dec, transform=ax2.get_transform('fk5'),
    #          ls='-', lw=1, label=my_label, color=color0)
    # # Finally we plot the streamer in velocity
    # ax3.plot(d_sky_au, v_lsr + vy1, label=my_label, color=color0)

# Plot legend at the end
# ax.legend()
ax2.legend(prop={'size': 8})
# ax3.legend()
plt.show()

if savekernel:
    vlsr_rad_kde_pickle = 'Velocity_Radius_KDE_reg_s.pickle'
    KDE_vel_rad = {'radius': xx, 'v_lsr': yy, 'dens': zz}
    with open(vlsr_rad_kde_pickle, 'wb') as f:
        pickle.dump(KDE_vel_rad, f)

if savemodel:
    stream_params = 'streamer_model_H2CO_0.39Msun_env_params.pickle'
    stream_model_params = {'theta0': theta0, 'r0': r0, 'phi0': phi0,
                           'v_r0': v_r0, 'omega0': omega0, 'v_lsr': v_lsr, 'inc': inc,
                           'PA': PA_ang}
    with open(stream_params, 'wb') as f:
        pickle.dump(stream_model_params, f)


    stream_pickle = 'streamer_model_H2CO_0.39Msun_env_vr.pickle'
    stream_model = {'ra': fil.ra.value*u.deg, 'dec': fil.dec.value*u.deg,
                    'd_sky_au': d_sky_au, 'vlsr': v_lsr + vy1}
    with open(stream_pickle, 'wb') as f:
        pickle.dump(stream_model, f)
