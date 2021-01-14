import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.patches import Ellipse, Rectangle

# We will load the strongest SO line we have (5_5-4_4) and H2CO
# Update = C+D conf.

SOfile = 'SO_55_44/Per-emb-50_C_l009l048_uvsub_SO_multi_integrated_python'
SO5645file = 'SO_56_45/Per-emb-50_C_l026l065_uvsub_SO_multi_integrated_python'
H2COfile = 'H2CO/Per-emb-50_C_l021l060_uvsub_H2CO_multi_integrated_python'
continuum = fits.getdata('SO_55_44/Per-emb-50_C_l009l048_cont.fits')[0]
rmsSO = 14.94e-3
rmsH2CO = 12.4e-3 # Jy/beam km/s
rmsSO5645 = 24.86e-3
SO = fits.getdata(SOfile+'.fits')
H2CO = fits.getdata(H2COfile+'.fits')
SO5645 = fits.getdata(SO5645file+'.fits')

header = fits.getheader(SOfile+'.fits')
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
im = ax.imshow(SO, cmap='inferno', vmin=0, vmax=0.6)
fig.colorbar(im, ax=ax, label='Jy/beam km/s')
ax.set_xlabel('RA (ICRS)')
ax.set_ylabel('DEC (ICRS)')
ax.set_xlim(zoomlims[0][1], zoomlims[0][0])
ax.set_ylim(zoomlims[1][0], zoomlims[1][1])


cntr1 = ax.contour(continuum, colors='r', levels=[0.007])
cntr2 = ax.contour(H2CO, colors='c', levels=np.array([3, 5, 7, 9, 11])*rmsH2CO)
cntr3 = ax.contour(SO5645, colors='m', levels=np.array([3, 5, 7, 9, 11])*rmsSO5645, linestyles='dashed')
# cntr4 = ax.contour(SO, colors='w', levels=np.array([3, 5, 7, 9, 11])*rmsSO, linestyles='dashed')
h1, _ = cntr1.legend_elements()
h2, _ = cntr2.legend_elements()
h3, _ = cntr3.legend_elements()
# h4, _ = cntr4.legend_elements()
# ax.legend([h1[0], h2[0], h3[0]], ['Continuum = 7 mJy/beam',
                                  # r'H$_2$CO(3$_{0,3}$- 2$_{0,2}$)', r'SO($5_5-4_4$)'], loc=0)
# ax.legend([h1[0], h2[0]], ['Continuum = 7 mJy/beam',
#                                   r'H$_2$CO(3$_{0,3}$- 2$_{0,2}$)'], loc=0)
ax.legend([h1[0], h2[0], h3[0]], ['Continuum = 7 mJy/beam',
                                  r'H$_2$CO(3$_{0,3}$- 2$_{0,2}$)', r'SO($5_6-4_5$)'], loc=0)
# ax.legend([h1[0], h2[0], h3[0], h4[0]], ['Continuum = 7 mJy/beam',
#                                   r'H$_2$CO(3$_{0,3}$- 2$_{0,2}$)', r'SO($5_6-4_5$)', r'SO($5_5-4_4$)'], loc=0)
ax.add_patch(beammarquee)
ax.add_patch(beam)
ax.add_artist(circle1)
# ax.scatter(markerpospix[0], markerpospix[1], s=200, c='red', marker='x', zorder=10)

# fig.savefig('comparison_H2CO_SO_55_44_2.eps', bbox_inches='tight')
