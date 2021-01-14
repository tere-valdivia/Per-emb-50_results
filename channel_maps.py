import aplpy
import numpy as np
import easy_aplpy
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

folder = 'SO_55_44/'
cubename = folder+'Per-emb-50_C_l009l048_uvsub_SO_multi'
cubehead = fits.getheader(cubename+'.fits')
wcs = WCS(cubehead)
wcsvel = wcs.sub([3])
continuum = 'SO_55_44/Per-emb-50_C_l009l048_cont.fits'
contlevel = 0.007
contcolor = 'red'
stretch = 'arcsinh'
velstart = 6
velend = 9
rows = 4
cols = 4
vmin = 0.001
vmax = 0.5
centerra = 52.2811737
centerdec = 31.3640438
radiusplot = 15
phasecent = [52.28236666667, 31.36586888889]

# plot the channel maps

velocities = np.linspace(velstart, velend, rows*cols)
chans = np.array([int(wcsvel.wcs_world2pix([p*1000], 0)[0][0]) for p in velocities])
easy_aplpy.settings.grid_label_color = 'white'
contourlist = [[[continuum, 0, [0.007], ['red']]]] * len(chans)
marker = [[[SkyCoord(phasecent[0], phasecent[1], unit=(u.deg, u.deg)), {'s': 16,
                                                                        'linewidth': 1.0, 'edgecolor': 'black', 'facecolor': 'black', 'marker': '+'}]]] * len(chans)
# SO 55-44
# marker = [[[SkyCoord(phasecent[0], phasecent[1], unit=(u.deg, u.deg)), {'linewidth': 1.0, 'edgecolor': 'black', 'facecolor': 'black', 'marker': '+'}],
#     [SkyCoord(52.2805875, 31.3628059, unit=(u.deg, u.deg)), {'linewidth': 1.0, 'edgecolor': 'black', 'facecolor': 'cyan', 'marker': '.'}],
#     [SkyCoord(52.2824167, 31.3660083, unit=(u.deg, u.deg)), {'linewidth': 1.0, 'edgecolor': 'black', 'facecolor': 'cyan', 'marker': '.'}],
#     [SkyCoord(52.2823333, 31.3656389, unit=(u.deg, u.deg)), {'linewidth': 1.0, 'edgecolor': 'black', 'facecolor': 'cyan', 'marker': '.'}]]] * len(chans)
# C180
# marker = [[[SkyCoord(phasecent[0], phasecent[1], unit=(u.deg, u.deg)), {'s': 16,
#                                                                         'linewidth': 1.0, 'edgecolor': 'black', 'facecolor': 'black', 'marker': '+'}],
#            [SkyCoord(52.2809582, 31.3663641, unit=(u.deg, u.deg)), {
#                'linewidth': 1.0, 'edgecolor': 'black', 'facecolor': 'cyan', 'marker': '.'}],
#            [SkyCoord(52.2809583, 31.3634283, unit=(u.deg, u.deg)), {
#                'linewidth': 1.0, 'edgecolor': 'black', 'facecolor': 'cyan', 'marker': '.'}],
#            [SkyCoord(52.2829880, 31.3619428, unit=(u.deg, u.deg)), {
#                'linewidth': 1.0, 'edgecolor': 'black', 'facecolor': 'cyan', 'marker': '.'}],
#            [SkyCoord(52.2832366, 31.3657628, unit=(u.deg, u.deg)), {
#                'linewidth': 1.0, 'edgecolor': 'black', 'facecolor': 'cyan', 'marker': '.'}]]] * len(chans)
# SO2 4_22 3_13
# marker = [[[SkyCoord(phasecent[0], phasecent[1], unit=(u.deg, u.deg)), {'s': 16,
#                                                                         'linewidth': 1.0, 'edgecolor': 'black', 'facecolor': 'black', 'marker': '+'}],
#            [SkyCoord(52.2822839, 31.3656213, unit=(u.deg, u.deg)), {
#                'linewidth': 1.0, 'edgecolor': 'black', 'facecolor': 'cyan', 'marker': '.'}],
#            [SkyCoord(52.2802127, 31.3622965, unit=(u.deg, u.deg)), {
#                'linewidth': 1.0, 'edgecolor': 'black', 'facecolor': 'cyan', 'marker': '.'}]]] * len(chans)

# easy_aplpy.plot.grid(cubename+'.fits', [rows, cols], chans, cmap='viridis',
#                      out=cubename+'_chanmap_'+stretch+'.png',
#                      vmin=vmin, vmax=vmax, stretch=stretch, recenter=[
#                      SkyCoord(centerra, centerdec, unit=(u.deg, u.deg)), radiusplot*u.arcsec],
#                      contours=contourlist, markers=marker)

easy_aplpy.plot.grid(cubename+'.fits', [rows, cols], chans, cmap='viridis',
                     out=cubename+'_chanmap_'+stretch+'.png',
                     vmin=vmin, vmax=vmax, stretch=stretch, recenter=[
                     SkyCoord(centerra, centerdec, unit=(u.deg, u.deg)), radiusplot*u.arcsec],
                     markers=marker)
