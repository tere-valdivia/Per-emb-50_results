"""
Author: Teresa Valdivia-Mena
Last revised August 30, 2022

This code is to extract the position-velocity diagrams for the emission cubes

Current state: SO2 (checked and ok)
"""

import sys
sys.path.append('../')
import regions
from astropy.visualization import simple_norm
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from spectral_cube import SpectralCube
from astropy.coordinates import SkyCoord
from astropy import units as u
from pvextractor import Path, extract_pv_slice, PathFromCenter
import numpy as np
from NOEMAsetup import *
import os

savepv = 1
saveplot = 0
vmin = 0.
vmax = 2
stretch = 'linear'
velinit = 0.01
velend = 14
radiusplot = 8/3600

# folder = '../'
# cubename = C18O_2_1_s
# folder = '../'
# cubename = SO_55_44_s_Jy
folder = '../'
cubename = SO2_11_1_11_10_0_10_s
cube = SpectralCube.read(folder+cubename+'.fits').with_spectral_unit(u.km /
                        u.s).spectral_slab(velinit*u.km/u.s, velend*u.km/u.s)
header = cube.header
ra = ra_Per50
dec = dec_Per50
anglepa = 170
plotname = folder+cubename+'_pvex_pvline_center_Per50_1arcsec_'+str(anglepa)+'PA_12arcsec_cutonly'
plotnamearcsec = folder+cubename+'_pvex_pvline_center_Per50_1arcsec_'+str(anglepa)+'PA_12arcsec_cutonly_arcsec'

# regionfile = folder + 'pvline.reg'

moment0 = cube.moment(order=0).hdu
wcsmom0 = WCS(moment0.header)

g_cent = SkyCoord(ra, dec)
length = 12 * u.arcsec

pa_angle_cent = anglepa * u.deg
path_cent = PathFromCenter(center=g_cent, length=length, angle=pa_angle_cent, width=1*u.arcsec)

patch_list = path_cent.to_patches(1, wcsmom0, alpha=0.3, color='w')

slice1 = extract_pv_slice(cube, path_cent)

if savepv:
    slice1.writeto(plotname+'.fits')

    if not os.path.exists(plotnamearcsec+'.fits'):
        pvdata = np.flip(slice1.data, axis=0) # so that the velocities grow instead of decreasing
        pvheader = slice1.header
        pvheader['CRPIX1'] = int(pvheader['NAXIS1']/2)
        pvheader['CDELT1'] = pvheader['CDELT1']*3600 # we just change the unit to be able to use it with Cote's code
        pvheader['CUNIT1'] = 'arcsec'
        if np.sign(pvheader['CDELT2'])<0:
            velinit = pvheader['CRVAL2'] + pvheader['CDELT2'] * (pvheader['NAXIS2'] - pvheader['CRPIX2']) # this is the velocity in the new first pixel
            pvheader['CRPIX2'] = 1 # the first pixel is our new reference
            pvheader['CRVAL2'] = velinit # the first pixel will have the lowest velocity
            pvheader['CDELT2'] *= -1
        hdu = fits.PrimaryHDU(data=pvdata, header=pvheader)
        hdu.writeto(plotnamearcsec+'.fits')

norm = simple_norm(slice1.data, stretch, min_cut=0, max_cut=0.15)

wcs = WCS(slice1.header)
len0 = slice1.header['NAXIS1']
offset0 = slice1.header['CRVAL1']
i0 = slice1.header['CRPIX1']
delta0 = slice1.header['CDELT1']
len1 = slice1.header['NAXIS2']
vel1 = slice1.header['CRVAL2']
i1 = slice1.header['CRPIX2']
delta1 = slice1.header['CDELT2']
offset = np.array([(offset0 + delta0*(i-i0+1))for i in range(len0)])*3600
velrange = np.array([(vel1 + delta1*(i-i1+1)) for i in range(len1)])/1.e3
zoomlims = wcsmom0.all_world2pix([ra.value-radiusplot, ra.value+radiusplot],
                                 [dec.value-radiusplot, dec.value+radiusplot], 0)

fig = plt.figure(figsize=(4, 4))

# ax = fig.add_subplot(211, projection=wcsmom0)
ax = fig.add_subplot(111, projection=wcsmom0)
norm_mom0 = simple_norm(moment0.data, 'asinh', asinh_a=0.3, min_cut=0)
im = ax.imshow(moment0.data, cmap='inferno', norm=norm_mom0)
# ax.scatter(pvline.start.x, pvline.start.y, color='g', label='(0,0)')
fig.colorbar(im, ax=ax, label=r'Integrated Intensity (Jy beam$^{-1}$ km s$^{-1}$)')
ax.set_xlabel('RA (J2000)')
ax.set_ylabel('DEC (J2000)')
for p in patch_list:
    ax.add_patch(p)
# ax.add_artist(artist)
# ax.legend()
ax.set_xlim(zoomlims[0][1], zoomlims[0][0])
ax.set_ylim(zoomlims[1][0], zoomlims[1][1])

if saveplot:
    fig.savefig(plotname+'.pdf', bbox_inches='tight')
