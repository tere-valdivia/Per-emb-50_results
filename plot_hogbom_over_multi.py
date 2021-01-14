
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.patches import Ellipse

# current state: SO_55_44

folder = 'SO_55_44/'
fitshog = folder+'Per-emb-50_C_l009l048_uvsub_SO_integrated'
fitsmulti = folder+'Per-emb-50_C_l009l048_uvsub_SO_multi_integrated'
continuum = fits.getdata('SO_55_44/Per-emb-50_C_l009l048_cont.fits')[0]
hogbom = fits.getdata(fitshog+'.fits')[0]
multi = fits.getdata(fitsmulti+'.fits')[0]
headerhog = fits.getheader(fitshog+'.fits')
headermulti = fits.getheader(fitsmulti+'.fits')
wcs = WCS(headerhog).celestial
ra = headerhog['RA']
dec = headerhog['DEC']
pixsize = headerhog['CDELT2']
scale = 1.896e-4*2/pixsize
phasecentpix = wcs.all_world2pix(ra, dec, 0)
radiusplot = 15/3600
radiusPB = 11.7/3600
bmaj = headerhog["BMAJ"]/pixsize
bmin = headerhog['BMIN']/pixsize
bpa = headerhog['BPA']
vmin = 0.05
vmax = 1.2
contlevels = [10**(-0.8), 10**(-0.6), 10**(-0.4), 10**(-0.2), 1]

zoomlims = wcs.all_world2pix([ra-radiusplot, ra+radiusplot],
                             [dec-radiusplot, dec+radiusplot], 0)


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection=wcs)
im = ax.imshow(multi, cmap='viridis', norm=colors.LogNorm(vmin=vmin, vmax=vmax))
fig.colorbar(im, ax=ax, label='Jy/beam km/s')
ax.set_xlim(zoomlims[0][1], zoomlims[0][0])
ax.set_ylim(zoomlims[1][0], zoomlims[1][1])
circle1 = plt.Circle(phasecentpix, radiusPB / pixsize, edgecolor='w', fill=False, linestyle='--')
ax.add_artist(circle1)
beam = Ellipse([zoomlims[0][1]+10, zoomlims[1][0]+10], bmaj,
               bmin, -bpa, facecolor='w', edgecolor='r')
ax.add_patch(beam)
cntr1 = ax.contour(hogbom, colors='k', linestyles='dotted',
                   levels=contlevels, label='Hogbom (this image)')
cntr2 = ax.contour(multi, colors='k', levels=contlevels, label='Multi')
cntr3 = ax.contour(continuum, colors='r', levels=[0.007])
h3, _ = cntr3.legend_elements()
h1, _ = cntr1.legend_elements()
h2, _ = cntr2.legend_elements()
ax.legend([h1[0], h2[0], h3[0]], ['Hogbom', 'Multi', 'Continuum = 7 mJy/beam'])
ax.set_xlabel('RA (ICRS)')
ax.set_ylabel('DEC (ICRS)')
ax.set_title('Comparison between Hogbom and Multi for SO $J=5_5-4_4$ (plotted: multi)')

textstr = 'Levels (Jy/beam km/s) = '
for level in contlevels:
    textstr += '\n'+str(round(level, 3))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
fontprops = fm.FontProperties(size=14)
scalebar = AnchoredSizeBar(ax.transData,
                           scale, '800 AU', 'lower right',
                           pad=0.1,
                           color='white',
                           frameon=False,
                           size_vertical=1,
                           fontproperties=fontprops)
ax.add_artist(scalebar)
fig.savefig(folder+'comparison.eps', bbox_inches='tight')
# gc = aplpy.FITSFigure('Per-emb-50_C_l009l048_uvsub_SO_integrated.fits', figure=fig)
# gc.show_colorscale(cmap='viridis', stretch='log')
# gc.recenter(52.2824163, 31.3658689, 15/3600)
# gc.add_colorbar()
# gc.colorbar.set_axis_label_text('$log_{10}$(Jy/beam km/s)')
# gc.show_contour('Per-emb-50_C_l009l048_uvsub_SO_multi_integrated.fits',
#                 levels=[10**(-0.8), 10**(-0.6), 10**(-0.4), 10**(-0.2), 1], colors='r', label='Multi')
# gc.show_contour('Per-emb-50_C_l009l048_uvsub_SO_integrated.fits', levels=[
#                 10**(-0.8), 10**(-0.6), 10**(-0.4), 10**(-0.2), 1], colors='k', linestyle='..', label='Hogbom (this data)')
