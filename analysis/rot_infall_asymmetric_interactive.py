'''
Author: Teresa Valdivia-Mena
Last revised August 31, 2022

This code is an interactive interface to calculate and plot the velocity versus
distance profiles, to further compare them against the position-velocity
diagram of SO emission, for the case there is an asymmetrical infall and
rotation model, for the redshifted and blueshifted sides.
'''

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.constants import G
from astropy.wcs import WCS
from astropy.io import fits
from astropy.visualization import simple_norm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from astropy.convolution import convolve, Gaussian1DKernel
from matplotlib.widgets import Slider, Button

import sys
sys.path.append('../')
from NOEMAsetup import *

### Important functions

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
M_0 = (1.7+0.58) * u.Msun #initial mass is star and disk
rc_0 = 250 * u.au # initial centrifugal radius is
offset_model = np.linspace(-1500, 1500, 300) * u.au


# pvfile = '../SO_55_44/CDconfig/pvex_Per-emb-50_CD_l009l048_uvsub_SO_multi_pbcor_pvline_center_Per50_1arcsec_170PA_12arcsec.fits'
pvfile = '../' + SO_55_44_PV + '.fits'
v_lsr = 7.5*u.km/u.s
arcsectoau = 293  # * u.au / u.arcsec
rms = 0.01
contourlevels = np.array([3,5,15,25]) * rms
vmin = 0
vmax = 0.45
inclination = 67 # 0 is face on
beamsize = 1.2 * u.arcsec


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

fig = plt.figure(figsize=(4,6))
plt.subplots_adjust(left=0.1, bottom=0.35)
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
ax.set_ylim([0, 15])
ax.set_xlim([-1500, 1500])
# Plot the initial curve

vel_redshift_0_nonconv = get_curve_redshift(offset_model, M_0, rc_0, inclination)
vel_blueshift_0_nonconv = get_curve_blueshift(offset_model, M_0, rc_0, inclination)
vel_redshift_0 = get_curve_redshift(offset_model, M_0, rc_0, inclination, smooth=True, kernelsize=beamstddev_pix)
vel_blueshift_0 = get_curve_blueshift(offset_model, M_0, rc_0, inclination, smooth=True, kernelsize=beamstddev_pix)

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

# Update the initial curve
# We create the sliders
deltaM0 = 0.1
deltarc0 = 10
axcolor = 'paleturquoise'
axM0_blue = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor=axcolor)
sM0_blue = Slider(axM0_blue, r'$M_{\star}$ blue', 0, 5, valinit=M_0.value, valstep=deltaM0)
axrc0_blue = plt.axes([0.2, 0.15, 0.6, 0.03], facecolor=axcolor)
src0_blue = Slider(axrc0_blue, r'$r_c$ blue', 100, 500, valinit=rc_0.value, valstep=deltarc0)
axM0_red = plt.axes([0.2, 0.2, 0.6, 0.03], facecolor=axcolor)
sM0_red = Slider(axM0_red, r'$M_{\star}$ red', 0, 5, valinit=M_0.value, valstep=deltaM0)
axrc0_red = plt.axes([0.2, 0.25, 0.6, 0.03], facecolor=axcolor)
src0_red = Slider(axrc0_red, r'$r_c$ red', 100, 500, valinit=rc_0.value, valstep=deltarc0)


def update(val):
    Mnew_blue = sM0_blue.val * u.Msun
    rcnew_blue = src0_blue.val * u.au
    Mnew_red = sM0_red.val * u.Msun
    rcnew_red = src0_red.val * u.au
    #get the new values in the complete range
    vel_redshift = get_curve_redshift(offset_model, Mnew_red, rcnew_red, inclination, smooth=True, kernelsize=beamstddev_pix)
    vel_blueshift = get_curve_blueshift(offset_model, Mnew_blue, rcnew_blue, inclination, smooth=True, kernelsize=beamstddev_pix)
    vel_redshift_nonconv = get_curve_redshift(offset_model, Mnew_red, rcnew_red, inclination)
    vel_blueshift_nonconv = get_curve_blueshift(offset_model, Mnew_blue, rcnew_blue, inclination)
    # cut off the excess
    vel_redshift = vel_redshift[np.where(np.abs(offset_model.value)<1000)]
    vel_blueshift = vel_blueshift[np.where(np.abs(offset_model.value)<1000)]
    vel_redshift_nonconv = vel_redshift_nonconv[np.where(np.abs(offset_model.value)<1000)]
    vel_blueshift_nonconv = vel_blueshift_nonconv[np.where(np.abs(offset_model.value)<1000)]
    # change the plots
    line_vel_red.set_ydata(vel_redshift+v_lsr.value)
    line_vel_blue.set_ydata(vel_blueshift+v_lsr.value)
    line_vel_red_nonconv.set_ydata(vel_redshift_nonconv+v_lsr.value)
    line_vel_blue_nonconv.set_ydata(vel_blueshift_nonconv+v_lsr.value)
    fig.canvas.draw_idle()

updateax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(updateax, 'Update', color=axcolor, hovercolor='0.975')
button.on_clicked(update)

plt.show()
