import aplpy
import numpy as np
import easy_aplpy
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

# Image
velfactor = 1000 # If velocity is not km/s, how much to multiply to have it
cubename = 'H2CO/Per-emb-50_C_l021l060_uvsub_H2CO_multi'
cubehead = fits.getheader(cubename+'.fits')
wcs = WCS(cubehead)
wcsvel = wcs.sub([3])
# continuum = '/Volumes/Elements/peremb50/SO_55_44/Per-emb-50_C_l009l048_cont.fits'
contlevel = 0.007
contcolor = 'red'
stretch = 'linear'
velstart = 6
velend = 10
rows = 5
cols = 4
vmin = 0.001
vmax = 0.11
centerra = 52.2811737
centerdec = 31.3640438
radiusplot = 15
phasecent = [52.28236666667, 31.36586888889]

velocities = np.linspace(velstart, velend, rows*cols)
chans = np.array([int(wcsvel.wcs_world2pix([p*velfactor], 0)[0][0]) for p in velocities])
easy_aplpy.settings.grid_label_color = 'white'


# contours
contourcubename = 'SO_55_44/Per-emb-50_C_l009l048_uvsub_SO_multi'
contourwcs = WCS(fits.getheader(contourcubename+'.fits'))
wcsvelcontour = contourwcs.sub([3])
wcsvelcontour.wcs_world2pix([], 0)
chanscontour = np.array([int(wcsvelcontour.wcs_world2pix([p*velfactor], 0)[0][0]) for p in velocities])
rmscontour = 14.94e-3
contourlist = [[[contourcubename+'.fits', chanscontour[j], [5*rmscontour, 10*rmscontour], ['red'], {'linewidths':0.5}]] for j in range(len(chanscontour))]

easy_aplpy.plot.grid(cubename+'.fits', [rows, cols], chans, cmap='inferno',
                     out='comparison_H2CO_image_with_SO_55_44_contour_'+stretch+'.png',
                     vmin=vmin, vmax=vmax, stretch=stretch, recenter=[
                     SkyCoord(centerra, centerdec, unit=(u.deg, u.deg)), radiusplot*u.arcsec],
                     contours=contourlist)
