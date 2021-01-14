import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.patches import Ellipse

# Current state: C18O

folder = 'SO_55_44/'
fitsfiles = ['Per-emb-50_C_l009l048_uvsub_SO_integrated',
             'Per-emb-50_C_l009l048_uvsub_SO_multi_integrated']
algorithms = ['Hogbom', 'Multi']
continuum = fits.getdata('SO_55_44/Per-emb-50_C_l009l048_cont.fits')[0]
vmin = 0.05
vmax = 1.2
vminlog = -2
vmaxlog = 1
fontprops = fm.FontProperties(size=14)
for fitsfile, alg in zip(fitsfiles, algorithms):
    data = fits.getdata(folder+fitsfile+'.fits')[0]
    header = fits.getheader(folder+fitsfile+'.fits')
    wcs = WCS(header).celestial
    ra = header['RA']
    dec = header['DEC']
    pixsize = header['CDELT2']
    scale = 1.896e-4*2/pixsize
    phasecentpix = wcs.all_world2pix(ra, dec, 0)
    radiusplot = 16/3600
    radiusPB = 11.7/3600
    bmaj = header["BMAJ"]/pixsize
    bmin = header['BMIN']/pixsize
    bpa = header['BPA']
    zoomlims = wcs.all_world2pix([ra-radiusplot, ra+radiusplot],
                                 [dec-radiusplot, dec+radiusplot], 0)
    circle1 = plt.Circle(phasecentpix, radiusPB /
                         header['CDELT2'], edgecolor='w', fill=False, linestyle='--')
    beam = Ellipse([zoomlims[0][1]+10, zoomlims[1][0]+10], bmaj,
                   bmin, -bpa, facecolor='w', edgecolor='r')

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(221, projection=wcs)
    im1 = ax.imshow(data, vmin=vmin, vmax=vmax)
    fig.colorbar(im1, ax=ax, label='Jy/beam km/s')
    ax.add_artist(circle1)
    ax.set_xlabel('RA (ICRS)')
    ax.set_xlabel('DEC (ICRS)')
    ax.set_title(alg)
    cntr1 = ax.contour(continuum, colors='r', levels=[0.007])
    h1, _ = cntr1.legend_elements()
    ax.legend([h1[0]], ['Continuum = 7 mJy/beam'], loc=3)

    circle2 = plt.Circle(phasecentpix, radiusPB /
                         header['CDELT2'], edgecolor='w', fill=False, linestyle='--')
    ax2 = fig.add_subplot(222, projection=wcs)
    im2 = ax2.imshow(np.log10(data), vmin=vminlog, vmax=vmaxlog)
    fig.colorbar(im2, ax=ax2, label='$log_{10}$(Jy/beam km/s)')
    ax2.add_artist(circle2)
    ax2.set_xlabel('RA (ICRS)')
    ax2.set_xlabel('DEC (ICRS)')
    ax2.set_title(alg+' log10 scale')
    cntr2 = ax2.contour(continuum, colors='r', levels=[0.007])
    scalebar = AnchoredSizeBar(ax2.transData,
                               scale*2, '800 AU', 'lower center',
                               pad=0.1,
                               color='white',
                               frameon=False,
                               size_vertical=2,
                               fontproperties=fontprops)
    ax2.add_artist(scalebar)

    circle3 = plt.Circle(phasecentpix, radiusPB /
                         header['CDELT2'], edgecolor='w', fill=False, linestyle='--')
    ax3 = fig.add_subplot(223, projection=wcs)
    im3 = ax3.imshow(data, vmin=vmin, vmax=vmax)
    ax3.set_xlim(zoomlims[0][1], zoomlims[0][0])
    ax3.set_ylim(zoomlims[1][0], zoomlims[1][1])
    fig.colorbar(im3, ax=ax3, label='Jy/beam km/s')
    ax3.add_artist(circle3)
    ax3.set_xlabel('RA (ICRS)')
    ax3.set_xlabel('DEC (ICRS)')
    ax3.set_title(alg+' zoom')
    ax3.add_patch(beam)
    cntr3 = ax3.contour(continuum, colors='r', levels=[0.007])

    circle4 = plt.Circle(phasecentpix, radiusPB /
                         header['CDELT2'], edgecolor='w', fill=False, linestyle='--')
    ax4 = fig.add_subplot(224, projection=wcs)
    im4 = ax4.imshow(np.log10(data), vmin=vminlog, vmax=vmaxlog)
    fig.colorbar(im4, ax=ax4, label='$log_{10}$(Jy/beam km/s)')
    ax4.set_xlim(zoomlims[0][1], zoomlims[0][0])
    ax4.set_ylim(zoomlims[1][0], zoomlims[1][1])
    ax4.add_artist(circle4)
    ax4.set_xlabel('RA (ICRS)')
    ax4.set_xlabel('DEC (ICRS)')
    ax4.set_title(alg+' log10 scale zoom')
    cntr4 = ax4.contour(continuum, colors='r', levels=[0.007])
    scale = 1.896e-4*2/header['CDELT2']
    fontprops = fm.FontProperties(size=14)
    scalebar = AnchoredSizeBar(ax4.transData,
                               scale, '400 AU', 'lower center',
                               pad=0.1,
                               color='white',
                               frameon=False,
                               size_vertical=1,
                               fontproperties=fontprops)
    ax4.add_artist(scalebar)
    beam = Ellipse([zoomlims[0][1]+10, zoomlims[1][0]+10], bmaj,
                   bmin, -bpa, facecolor='w', edgecolor='r')
    ax4.add_patch(beam)

    # fig.savefig(folder+fitsfile+'_plot.eps', bbox_inches='tight')
