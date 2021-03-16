import sys
sys.path.append('../')

import numpy as np
import astropy.units as u
import velocity_tools.stream_lines as SL
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord, FK5
from scipy import stats
from NOEMAsetup import *
import aplpy
from mpl_toolkits.mplot3d import Axes3D
import pickle


'''
Personal notes:
The model is done based on the best fitted v_lsr of the streamer, not
on the intensity image

Values of inclination and PA first taken from Segura-Cox et al 2016
Agurto-Gangas et al 2019:
- M_env = 2.2 Msun (Upper estimate of mass assumes T_d = 20K)
- M_disk = 0.58 Msun
Fiorellino et al 2021: M_s = 1.71 Msun (mean value for an age=1Myr)

Need to constrain the envelope mass.
- modelname = 'envelope_2Msun_highvel':
theta0 = 76.*u.deg  # rotate clockwise
r0 = 3272.*u.au
phi0 = 15.*u.deg  # rotate the plane
v_r0 = 3.5*u.km/u.s
omega0 = 8.e-13/u.s
-

We test 2 streamer regions. Region streamer_region_l.reg does not include the
"kink"
'''

savemodel = True
savekernel = False

modelname = 'envelope_018Msun_midvel'
regionfile = 'data/region_streamer_l.reg'

# Main parameters to generate a streamline
M_s = 1.71*u.Msun # was 2.9
M_env = 0.18*u.Msun # was 2.2
# M_env = 2.2*u.Msun
M_disk = 0.58*u.Msun
Mstar = (M_s+M_env+M_disk)
inc = (360-(90-67))*u.deg  # should be almost edge on
PA_ang = -(170-90)*u.deg
v_lsr = 7.48*u.km/u.s  # +- 0.14 km/s according to out C18O data


# Create Per-emb-50 reference coordinate system

Per50_c = SkyCoord(ra_Per50, dec_Per50, frame='fk5')
Per50_ref = Per50_c.skyoffset_frame()
freq_H2CO_303_202 = fits.getheader('../'+H2CO_303_202_TdV_s+'.fits')['RESTFREQ'] * u.Hz

# Obtain the offset radius from Per-emb-50 and the v_lsr for each
# valid pixel in the streamer
r_proj, v_los = per_emb_50_get_vc_r('../'+H2CO_303_202_fit_Vc+'.fits', '../'+regionfile)

# create the grid for the kernel distribution
xmin = 0
xmax = 3500
ymin = 6.
ymax = 8.
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
#
gd_vlos = np.isfinite(r_proj*v_los)  # filter nan
values = np.vstack([r_proj[gd_vlos].value, v_los[gd_vlos].value])

kernel = stats.gaussian_kde(values)
zz = np.reshape(kernel(positions).T, xx.shape)
zz /= zz.max()  # normalization of probability

# If you want to load a previously saved model
#
# pickle_in = open('streamer_model'+str(modelnum)+'_params.pickle', "rb")
# paramdict = pickle.load(pickle_in)
# # print(paramdict)
# paramsload = []
# for key in paramdict:
#     paramsload.append(paramdict[key])


# Diagnostic plots:

# For my understanding of the axis
fig3d = plt.figure()
ax3d = fig3d.gca(projection='3d')
# ax3d.scatter(x1[0].value, y1[0].value, z1[0].value, color='k')
ax3d.set_xlabel('x')
ax3d.set_ylabel('y')
ax3d.set_zlabel('z')
# plt.show()

# Distribution of points and kernel
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111)
ax.set_xlabel('Projected distance (au)')
ax.set_ylabel(r"V$_{lsr}$ (km s$^{-1}$)")
ax.contourf(xx, yy, zz, cmap='Greys', levels=np.arange(0.1, 1.2, 0.1), vmin=0., vmax=1.1)
ax.set_ylim([6,8])
ax.set_xlim([0, 3500])

if savekernel:
    fig.savefig('streamline_vlsr_r_kde_peremb50_H2CO_reg.pdf')

# plt.show()


# Streamer over intensity map

fig2 = aplpy.FITSFigure('../'+H2CO_303_202_TdV_s+'.fits', figsize=(4, 4))
fig2.show_grayscale(vmin=0, vmax=3., invert=True)
fig2.add_colorbar()
setup_plot_noema(fig2, label_col='black', star_col='yellow')
# fig2.show_regions('../'+regionfile)
fig2.show_circles(ra_Per50, dec_Per50, pb_noema(freq_H2CO_303_202).to(u.deg)*0.5,
                  ls=':', color='black')

# fig2.add_grid()
# plt.show()


# Calculate and plot streamline

# Define the different initial theta and phi to generate the stream lines
# Currently in testing
theta0 = 80.*u.deg  # rotate clockwise
r0 = 3330.*u.au
phi0 = 16.5*u.deg  # rotate the plane
v_r0 = 1.1*u.km/u.s
omega0 = 3.23e-13/u.s

(x1, y1, z1), (vx1, vy1, vz1) = SL.xyz_stream(
    mass=Mstar, r0=r0, theta0=theta0, phi0=phi0,
    omega=omega0, v_r0=v_r0, inc=inc, pa=PA_ang, rmin=300*u.au)
r_c = SL.r_cent(Mstar, omega0, r0)
d_sky_au = np.sqrt(x1**2 + z1**2)
# Stream line into arcsec
dra_stream = -x1.value / dist_Per50  # ask why it is negative, but probably
# because the east is to the left of the plot
ddec_stream = z1.value / dist_Per50
fil = SkyCoord(dra_stream*u.arcsec, ddec_stream*u.arcsec,
               frame=Per50_ref).transform_to(FK5)

ax.plot(d_sky_au, v_lsr + vy1, color='red', label='Streamline')
ax.axhline(v_lsr.value, color='k', label=r'Per-emb-50 $v_{lsr}$', ls=':')
ax.annotate(r'$r_c = {}$'.format(np.round(r_c,0)), (0.5, 0.2), xycoords='axes fraction', size=14)
ax.legend(loc=(0.45, 0.75), frameon=False, fontsize=15)
ax3d.scatter(x1[0].value, y1[0].value, z1[0].value, color='k')
ax3d.plot(x1.value, y1.value, z1.value, color='red')
fig2.show_markers(fil.ra.value*u.deg, fil.dec.value*u.deg,
                  marker='o', color='red', s=1)
if savemodel:
    # fig.savefig('streamline_vlsr_r_kde_peremb50_H2CO_'+modelname+'.pdf', dpi=300, bbox_inches='tight')
    # fig2.savefig('streamline_image_peremb50_H2CO_'+modelname+'.pdf', dpi=300)
    stream_params = 'streamer_'+modelname+'_params.pickle'
    stream_model_params = {'theta0': theta0, 'r0': r0, 'phi0': phi0,
                           'v_r0': v_r0, 'omega0': omega0, 'v_lsr': v_lsr}
    with open(stream_params, 'wb') as f:
        pickle.dump(stream_model_params, f)
    stream_pickle = 'streamer_'+modelname+'.pickle'
    stream_model = {'ra': fil.ra.value*u.deg, 'dec': fil.dec.value*u.deg,
                    'd_sky_au': d_sky_au, 'vlsr': v_lsr + vy1}
    with open(stream_pickle, 'wb') as f:
        pickle.dump(stream_model, f)

# theta0, r0, phi0, v_r0, omega0, v_lsr = paramsload
# omega0 = 4.e-13 / u.s
# thetalist = [0*u.deg, 30*u.deg, 60*u.deg, 90*u.deg]
# r0 = 1500 * u.au
# phi0 = 90*u.deg
# omegalist = [1.e-13 / u.s,2.e-13 / u.s, 3.e-13 / u.s, 4.e-13 / u.s, 8.e-13 / u.s]
#
# for theta0 in thetalist:
#     (x1, y1, z1), (vx1, vy1, vz1) = SL.xyz_stream(
#         mass=Mstar, r0=r0, theta0=theta0, phi0=phi0,
#         omega=omega0, v_r0=v_r0, inc=inc, pa=PA_ang, rmin=200*u.au)
#
#     d_sky_au = np.sqrt(x1**2 + z1**2)
#     # Stream line into arcsec
#     dra_stream = x1.value / dist_Per50  # ask why it is negative, but probably
#     # because the east is to the left of the plot
#     ddec_stream = z1.value / dist_Per50
#     fil = SkyCoord(dra_stream*u.arcsec, ddec_stream*u.arcsec,
#                    frame=Per50_ref).transform_to(FK5)
#
#     fig2.show_circles(ra_Per50, dec_Per50, pb_noema(freq_H2CO_303_202).to(u.deg)*0.5,
#                       ls=':', color='black')
#     ax.plot(d_sky_au, v_lsr + vy1) #, color='red')
#     ax3d.scatter(x1[0].value, y1[0].value, z1[0].value, color='k')
#     ax3d.plot(x1.value, y1.value, z1.value) #, color='red')
#     fig2.show_markers(fil.ra.value*u.deg, fil.dec.value*u.deg,
#                       marker='o', color='red', s=1)

plt.show()
#
# if savekernel:
#     vlsr_rad_kde_pickle = 'Velocity_Radius_KDE_reg.pickle'
#     KDE_vel_rad = {'radius': xx, 'v_lsr': yy, 'dens': zz}
#     with open(vlsr_rad_kde_pickle, 'wb') as f:
#         pickle.dump(KDE_vel_rad, f)
#
