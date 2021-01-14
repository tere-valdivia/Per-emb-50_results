import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.patches import Ellipse, Rectangle

# We will load the strongest SO line we have (5_5-4_4) and H2CO
plotname = 'comparison_SO_55_44_CandDconfiguration.eps'
imagefile = 'SO_55_44/CDconfig/Per-emb-50_CD_l009l048_uvsub_SO_multi_integrated_python'
contourfile1 = imagefile
contourfile2 = 'SO_55_44/Per-emb-50_C_l009l048_uvsub_SO_multi_integrated_python'
# continuum = fits.getdata('SO_55_44/Per-emb-50_C_l009l048_cont.fits')[0]
rmsimagefile = 26.82e-3
rmscontourfile1 = 26.82e-3 # Jy/beam km/s
rmscontourfile2 = 27.03e-3
image = fits.getdata(imagefile+'.fits')
contour1 = fits.getdata(contourfile1+'.fits')
contour2 = fits.getdata(contourfile2+'.fits')
vmin = 0.0
vmax = 0.5

labelcontour1 = 'C+D'
labelcontour2 = 'C'

###################

header = fits.getheader(imagefile+'.fits')
wcs = WCS(header).celestial

ra = header['RA']
dec = header['DEC']
pixsize = header['CDELT2']
radiusplot = 16/3600
radiusPB = 11.7/3600
bmaj = header["BMAJ"]/pixsize
bmin = header['BMIN']/pixsize
bpa = header['BPA']

# Contour Plot
phasecentpix = wcs.all_world2pix(ra, dec, 0)
markerpospix = wcs.all_world2pix(52.2819472, 31.3654766, 0)

zoomlims = wcs.all_world2pix([ra-radiusplot, ra+radiusplot],
                             [dec-radiusplot, dec+radiusplot], 0)
circle1 = plt.Circle(phasecentpix, radiusPB /
                     header['CDELT2'], edgecolor='w', fill=False, linestyle='--')
beam = Ellipse([zoomlims[0][1]+15, zoomlims[1][0]+15], bmaj,
               bmin, -bpa, facecolor='k', edgecolor='k')
beammarquee = Rectangle([zoomlims[0][1], zoomlims[1][0]], 30, 30, facecolor='w')

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection=wcs)
im = ax.imshow(image, cmap='inferno', vmin=vmin, vmax=vmax)
fig.colorbar(im, ax=ax, label='Jy/beam km/s')
ax.set_xlabel('RA (ICRS)')
ax.set_ylabel('DEC (ICRS)')
ax.set_xlim(zoomlims[0][1], zoomlims[0][0])
ax.set_ylim(zoomlims[1][0], zoomlims[1][1])


cntr1 = ax.contour(contour1, colors='r', levels=np.array([3, 5, 7, 9, 11])*rmscontourfile1, zorder=9)
cntr2 = ax.contour(contour2, colors='c', levels=np.array([3, 5, 7, 9, 11])*rmscontourfile2)
# cntr3 = ax.contour(SO5645, colors='m', levels=np.array([3, 5, 7, 9, 11])*rmsSO5645, linestyles='dashed')
# cntr4 = ax.contour(SO, colors='w', levels=np.array([3, 5, 7, 9, 11])*rmsSO, linestyles='dashed')
h1, _ = cntr1.legend_elements()
h2, _ = cntr2.legend_elements()
# h3, _ = cntr3.legend_elements()
# h4, _ = cntr4.legend_elements()
# ax.legend([h1[0], h2[0], h3[0]], ['Continuum = 7 mJy/beam',
                                  # r'H$_2$CO(3$_{0,3}$- 2$_{0,2}$)', r'SO($5_5-4_4$)'], loc=0)
ax.legend([h1[0], h2[0]], [labelcontour1, labelcontour2], loc=0)
# ax.legend([h1[0], h2[0], h3[0]], ['Continuum = 7 mJy/beam',
#                                   r'H$_2$CO(3$_{0,3}$- 2$_{0,2}$)', r'SO($5_6-4_5$)'], loc=0)
# ax.legend([h1[0], h2[0], h3[0], h4[0]], ['Continuum = 7 mJy/beam',
#                                   r'H$_2$CO(3$_{0,3}$- 2$_{0,2}$)', r'SO($5_6-4_5$)', r'SO($5_5-4_4$)'], loc=0)
ax.add_patch(beammarquee)
ax.add_patch(beam)
ax.add_artist(circle1)
# ax.scatter(markerpospix[0], markerpospix[1], s=200, c='red', marker='x', zorder=10)
ax.scatter(phasecentpix[0], phasecentpix[1], s=100, c='k', marker='x', zorder=10)

fig.savefig(plotname, bbox_inches='tight')
