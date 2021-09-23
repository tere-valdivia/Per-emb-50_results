import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.constants import G
from astropy.wcs import WCS
from astropy.io import fits
from astropy.visualization import simple_norm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

######################
# General functions and constants
######################

# where we modify mass and rc
M = (1.7+0.58+0.39) * u.Msun
# M_list = np.array([0.5, 1, 1.5]) * u.Msun
# r_border = 50 * u.au
# change mass
# rc = r_border * 2 # centrifugal radius, cent. barrier is 1/2 this
# rc = 240 * u.au
rc_list = np.array([200, 240, 300]) * u.au

# where we modify centrifugal barrier and maximum rotational velocity
# vrotmax_list = [1.5, 3.5, 5.5] * u.km/u.s
# vrotmax = 4.5 * u.km/u.s
# rc = 240 * u.au
# rc_list = np.array([100, 200, 300]) * u.au

def v_inf(radius, l, M):
    # l = L/m
    vel = np.sqrt(2 * G * M / radius - l**2 /(radius**2))
    index_rep = np.where(radius.value<rc.value/2)
    vel[index_rep] = np.nan
    return vel.to(u.km/u.s)

def v_rot(radius, l):
    vel = l/radius
    index_rep = np.where(radius.value<rc.value/2)
    vel[index_rep] = np.nan
    return vel.to(u.km/u.s)

def v_proj(x, y, l, M):
    r = np.sqrt(x**2 + y**2)
    tot = v_rot(r, l) * x/r + v_inf(r, l, M) * y/r
    return tot

######################
# PV image loading and plotting
######################
pvfile = '../SO_55_44/CDconfig/pvex_Per-emb-50_CD_l009l048_uvsub_SO_multi_pbcor_pvline_center_Per50_1arcsec_170PA_12arcsec.fits'
# pvfile = '../C18O/CDconfig/JEP/position_velocity/pvex_JEP_mask_multi_Per-emb-50_CD_l025l064_uvsub_C18O_pvline_center_Per50_1arcsec_170PA_12arcsec.fits'
# savename = 'PV_diagram_SO_170_with_Sakai_toymodel_length1600AU_testvmac_constrc.pdf'
# savename = 'PV_diagram_C18O_170_with_Sakai_toymodel_length1600AU.pdf'

v_lsr = 7.48*u.km/u.s
arcsectoau = 293  # * u.au / u.arcsec
pvdata = fits.getdata(pvfile)
pvheader = fits.getheader(pvfile)
rms = 0.01
contourlevels = np.array([3,5,15,25]) * rms
# contourlevels = np.array([3,5]) * rms
vmin = 0
vmax = 0.45
inclination = 67 # 0 is face on
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

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
norm = simple_norm(pvdata, 'linear', min_cut=vmin,max_cut=vmax)
pcolor = ax.pcolor(offset.value, vel.value, pvdata, shading='auto', norm=norm, cmap='Blues')
contours = ax.contour(offset.value, vel.value, pvdata, contourlevels, colors='k', linewidths=0.5)
fig.colorbar(pcolor, ax=ax, label=r'Intensity (Jy beam$^{-1}$)')

######################
# Sakai et al 2014 toy model implementation
######################
x = np.linspace(0, 1600, 300) * u.au
x_neg = np.linspace(0, -1600, 300) * u.au


colors = ['green','red', 'orange']
for rc, color in zip(rc_list, colors):
    # when we change mass and rc
    l_test = np.sqrt(rc * G * M).to(u.km**2/u.s)
    # when we change vmax and rc
    # l_test = (rc/2 * vrotmax).to(u.km**2/u.s)
    # M = (rc *vrotmax**2 / (2*G)).to(u.Msun)
    y_sample = np.linspace(rc.value/2, 1600, 50) * u.au # this one is important that starts at the rc/2
    vel_total = [] # for positive
    for y in y_sample:
        vel = v_proj(x, y, l_test, M) * np.sin(inclination*np.pi/180)
        vel_total.append(vel)

    vel_neg = []
    for y in y_sample:
        vel = v_proj(x_neg, y, l_test, M) * np.sin(inclination*np.pi/180)
        vel_neg.append(vel)

    velocity_max = np.nanmax(vel_total, axis=0)
    vel_negx_max = np.nanmax(vel_neg, axis=0)
    velocity_negx_min = -velocity_max
    velocity_min = -vel_negx_max

    ax.plot(x.value, velocity_max+v_lsr.value, color=color)
    ax.plot(x_neg.value, vel_negx_max+v_lsr.value, color=color)
    ax.plot(x_neg.value, velocity_negx_min+v_lsr.value, color=color)
    ax.plot(x.value, velocity_min+v_lsr.value, color=color, label='rc = '+str(rc))
ax.axhline(v_lsr.value,color='k', linestyle=':', linewidth=1)
ax.axvline(0, color='k', linestyle=':', linewidth=1)
ax.set_ylabel(r'$v_{LSR}$ (km s$^{-1}$)')
ax.set_xlabel('Offset distance (au)')
plt.legend(prop={'size': 6})
# fig.savefig(savename, bbox_inches='tight', dpi=300)
