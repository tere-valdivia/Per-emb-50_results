import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.constants import G
from astropy.wcs import WCS
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.convolution import convolve, Gaussian1DKernel
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.widgets import Slider, Button
# TODO: add convolution

def v_inf(radius, l, M, rc):
    # l = L/m
    vel = np.sqrt(2 * G * M / radius - l**2 /(radius**2))
    index_rep = np.where(radius.value<rc.value/2)
    vel[index_rep] = np.nan
    return vel.to(u.km/u.s)

def v_rot(radius, l, rc):
    vel = l/radius
    index_rep = np.where(radius.value<rc.value/2)
    vel[index_rep] = np.nan
    return vel.to(u.km/u.s)

def v_proj(x, y, l, M, rc):
    r = np.sqrt(x**2 + y**2)
    tot = v_rot(r, l, rc) * x/r + v_inf(r, l, M, rc) * y/r
    return tot


#### Parameters
M_red= 4 * u.Msun #initial mass is star and disk
rc_red = 130 * u.au # initial centrifugal radius is
M_blue = 2.9 * u.Msun #initial mass is star and disk
rc_blue = 100 * u.au # initial centrifugal radius is
offset_model = np.linspace(-1500, 1500, 300) * u.au

savename = 'PV_diagram_SO_170_with_Sakai_toymodel_asymetric_vlsr7.5.pdf'
pvfile = '../SO_55_44/CDconfig/pvex_Per-emb-50_CD_l009l048_uvsub_SO_multi_pbcor_pvline_center_Per50_1arcsec_170PA_12arcsec.fits'
v_lsr = 7.5*u.km/u.s
arcsectoau = 293  # * u.au / u.arcsec
beamsize = 1.2 * u.arcsec
rms = 0.01
contourlevels = np.array([3,5,15,25]) * rms
vmin = 0
vmax = 0.45
inclination = 67 # 0 is face on


# Calculating the curve
def get_curve_redshift(x, M, rc, inc, smooth=False, kernelsize = None):
    h0 = np.sqrt(rc * G * M).to(u.km**2/u.s)
    y_sample = np.linspace(rc.value/2, np.amax(x.value), 50) * u.au # this one is important that starts at the rc/2
    vel_total = []
    for y in y_sample:
        vel = v_proj(x, y, h0, M, rc) * np.sin(inc*np.pi/180)
        vel_total.append(vel)

    velocity_max = np.nanmax(vel_total, axis=0)
    if smooth:
        gauss_kernel = Gaussian1DKernel(kernelsize)
        smoothed_velocity_max = convolve(velocity_max, gauss_kernel)
    return velocity_max if not smooth else smoothed_velocity_max

def get_curve_blueshift(x, M, rc, inc, smooth=False, kernelsize = None):
    velmax = get_curve_redshift(x, M, rc, inc, smooth, kernelsize)
    velocity_max = -np.flip(velmax)
    return velocity_max

# prepare the plot

pvdata = fits.getdata(pvfile)
pvheader = fits.getheader(pvfile)

delta0 = pvheader['CRVAL1']
delta_delta = pvheader['CDELT1']
delta_pix0 = pvheader['CRPIX1']
delta_npix = pvheader['NAXIS1']
vel0 = pvheader['CRVAL2']
delta_vel = pvheader['CDELT2']
vel_pix0 = pvheader['CRPIX2']
vel_npix = pvheader['NAXIS2']

beamstddev_pix = beamsize.to(u.deg).value / delta_delta / 2.35
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
# ax.set_autoscale_on(False)
ax.set_ylabel(r'$v_{LSR}$ (km s$^{-1}$)')
ax.set_xlabel('Offset distance (au)')
ax.axhline(v_lsr.value,color='k', linestyle=':', linewidth=1)
ax.axvline(0, color='k', linestyle=':', linewidth=1)
ax.set_ylim([0, 14])
ax.set_xlim([-1500, 1500])
ax.set_xticks([-1000, -500, 0, 500, 1000])

vel_redshift_0_nonconv = get_curve_redshift(offset_model, M_red, rc_red, inclination)
vel_blueshift_0_nonconv = get_curve_blueshift(offset_model, M_blue, rc_blue, inclination)
vel_redshift_0 = get_curve_redshift(offset_model, M_red, rc_red, inclination, smooth=True, kernelsize=beamstddev_pix)
vel_blueshift_0 = get_curve_blueshift(offset_model, M_blue, rc_blue, inclination, smooth=True, kernelsize=beamstddev_pix)

offset_model_plot = offset_model[np.where(np.abs(offset_model.value)<1000)]
vel_redshift_plot = vel_redshift_0[np.where(np.abs(offset_model.value)<1000)]
vel_blueshift_plot = vel_blueshift_0[np.where(np.abs(offset_model.value)<1000)]
vel_redshift_nonconv_plot = vel_redshift_0_nonconv[np.where(np.abs(offset_model.value)<1000)]
vel_blueshift_nonconv_plot = vel_blueshift_0_nonconv[np.where(np.abs(offset_model.value)<1000)]

line_vel_red, = ax.plot(offset_model_plot.value, vel_redshift_plot+v_lsr.value, color='red', label='Convolved')
line_vel_blue, = ax.plot(offset_model_plot.value, vel_blueshift_plot+v_lsr.value, color='blue')
line_vel_red_nonconv, = ax.plot(offset_model_plot.value, vel_redshift_nonconv_plot+v_lsr.value, color='red', ls=':', label='Not convolved')
line_vel_blue_nonconv, = ax.plot(offset_model_plot.value, vel_blueshift_nonconv_plot+v_lsr.value, color='blue', ls=':')
bar_transparent = ax.axvspan(-1500, np.amin(offset_model_plot.value), alpha=0.5, edgecolor='grey', facecolor='grey')
bar_transparent2 = ax.axvspan(np.amax(offset_model_plot.value), 1500, alpha=0.5, edgecolor='grey', facecolor='grey')

ax.legend(loc=4, prop={'size': 6})
fig.savefig(savename, bbox_inches='tight', dpi=300)
