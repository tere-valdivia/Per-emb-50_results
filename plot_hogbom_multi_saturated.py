import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.patches import Ellipse, Rectangle

# Always remember the PB depends on the line!
folder = 'H2CO/'
fitsfiles = ['Per-emb-50_C_l021l060_uvsub_H2CO_integrated_python',
             'Per-emb-50_C_l021l060_uvsub_H2CO_multi_integrated_python']
# folder = 'SO_55_44/'
# fitsfiles = ['Per-emb-50_C_l009l048_uvsub_SO_integrated',
#              'Per-emb-50_C_l009l048_uvsub_SO_multi_integrated']
algorithms = ['Hogbom', 'Multi']
continuum = fits.getdata('SO_55_44/Per-emb-50_C_l009l048_cont.fits')[0]
satlevel = 'saturated'


# SO 55-44:
# vmin = -0.2
# vmax = 1.2
# vmax = 0.1

# # C18O 2-1:
# vmin = -0.1
# # vmax = 0.18
# vmax = 0.05

# SO2 4_22_3_13:
# vmin = -0.02
# vmax = 0.19
# vmax = 0.05

# H2Co:
vmin = 0.0
# # vmax = 0.18
vmax = 0.03


fig = plt.figure(figsize=(12, 6))

for fitsfile, alg, i in zip(fitsfiles, algorithms, range(len(algorithms))):
    data = fits.getdata(folder+fitsfile+'.fits')
    header = fits.getheader(folder+fitsfile+'.fits')
    wcs = WCS(header).celestial
    ra = header['RA']
    dec = header['DEC']
    pixsize = header['CDELT2']
    phasecentpix = wcs.all_world2pix(ra, dec, 0)
    radiusplot = 25/3600
    radiusPB = 11./3600
    bmaj = header["BMAJ"]/pixsize
    bmin = header['BMIN']/pixsize
    bpa = header['BPA']
    zoomlims = wcs.all_world2pix([ra-radiusplot, ra+radiusplot],
                                 [dec-radiusplot, dec+radiusplot], 0)
    circle1 = plt.Circle(phasecentpix, radiusPB /
                         header['CDELT2'], edgecolor='w', fill=False, linestyle='--')
    beam = Ellipse([zoomlims[0][1]+15, zoomlims[1][0]+15], bmaj,
                   bmin, -bpa, facecolor='k', edgecolor='k')
    beammarquee = Rectangle([zoomlims[0][1], zoomlims[1][0]], 30, 30, facecolor='w')

    ax = fig.add_subplot(1, 2, i+1, projection=wcs)
    im = ax.imshow(data, cmap='afmhot', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label='Jy/beam km/s')
    ax.add_artist(circle1)
    ax.set_xlabel('RA (ICRS)')
    ax.set_ylabel('DEC (ICRS)')
    ax.set_title(alg)
    ax.set_xlim(zoomlims[0][1], zoomlims[0][0])
    ax.set_ylim(zoomlims[1][0], zoomlims[1][1])
    ax.add_patch(beammarquee)
    ax.add_patch(beam)

    cntr1 = ax.contour(continuum, colors='r', levels=[0.007])
    if i == 0:
        h1, _ = cntr1.legend_elements()
        ax.legend([h1[0]], ['Continuum = 7 mJy/beam'], loc=3)


# fig.savefig(folder+'hogbom_multi_'+satlevel+'.eps', bbox_inches='tight')

fig.savefig(folder+'comparison_'+satlevel+'.eps', bbox_inches='tight')
