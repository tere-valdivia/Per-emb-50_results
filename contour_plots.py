import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.patches import Ellipse, Rectangle

#####################

# filename = 'SO2_12_39_12_210/Per-emb-50_C_l046l085_uvsub_SO2_multi_integrated_python'
filenames = ['SO2_12_39_12_210/Per-emb-50_C_l046l085_uvsub_SO2_multi_integrated_python','SO2_4_22_3_13/Per-emb-50_C_l042l081_uvsub_SO2_multi_integrated_python','SO2_11_1_11_10_0_10/Per-emb-50_C_l031l070_uvsub_SO2_multi_integrated_python']
rmslist = [16.49e-3, 14.52e-3, 12.39e-3]
lines = [r'Per-emb-50 SO$_2$($12_{3,9}-12_{2,10}$), rms='+str(round(rmslist[0]*1e3,2))+r' mJy/beam km/s', r'Per-emb-50 SO$_2$($4_{2,2}-3_{1,3}$), rms='+str(round(rmslist[1]*1e3,2))+r' mJy/beam km/s', r'Per-emb-50 SO$_2$($11_{1,11}-10_{0,10}$), rms='+str(round(rmslist[2]*1e3,2))+r' mJy/beam km/s']

plotname = 'comparison_SO2_positions_3_5sigma.eps'
colorlist = ['red', 'green','blue']
contourlist = []
radiusplot = 16/3600
radiusPB = 21.76/2/3600
logscale = False

#####################


fig = plt.figure(figsize=(8, 9))
# In the case they had different wcs, you need to reproject!
header = fits.getheader(filenames[0]+'.fits')
wcs = WCS(header).celestial
ax = fig.add_subplot(111, projection=wcs)
ra = header['RA']
dec = header['DEC']
pixsize = header['CDELT2']
bmaj = header["BMAJ"]/pixsize
bmin = header['BMIN']/pixsize
bpa = header['BPA']
# continuum = fits.getdata('SO_55_44/Per-emb-50_C_l009l048_cont.fits')[0]
phasecentpix = wcs.all_world2pix(ra, dec, 0)
markerpospix = wcs.all_world2pix(52.2823389, 31.3657327, 0)
markerpospix2 = wcs.all_world2pix(52.2821863, 31.3654721, 0)
markerpospix3 = wcs.all_world2pix(52.2803453, 31.3623526, 0)
zoomlims = wcs.all_world2pix([ra-radiusplot, ra+radiusplot],
                             [dec-radiusplot, dec+radiusplot], 0)

for filename, line, rms, col in zip(filenames, lines, rmslist, colorlist):
    data = fits.getdata(filename+'.fits')
    contourlist.append(ax.contour(data, levels=[3*rms, 5*rms], label=line, colors=col))
    beam = Ellipse([zoomlims[0][1]+15, zoomlims[1][0]+15], bmaj,
                   bmin, -bpa, facecolor='k', edgecolor='k')
    beammarquee = Rectangle([zoomlims[0][1], zoomlims[1][0]], 30, 30, facecolor='w')
    ax.add_patch(beammarquee)
    ax.add_patch(beam)

circle1 = plt.Circle(phasecentpix, radiusPB /
                     header['CDELT2'], edgecolor='k', fill=False, linestyle='--')
ax.add_artist(circle1)
ax.set_xlabel('RA (ICRS)')
ax.set_ylabel('DEC (ICRS)')
ax.scatter(markerpospix[0], markerpospix[1], c='k', s=60, marker='x')
ax.scatter(markerpospix2[0], markerpospix2[1], c='k', s=60, marker='x')
ax.scatter(markerpospix3[0], markerpospix3[1], c='k', s=60, marker='x')
ax.scatter(phasecentpix[0], phasecentpix[1], c='k', s=40, marker='o')
ax.set_xlim(zoomlims[0][1], zoomlims[0][0])
ax.set_ylim(zoomlims[1][0], zoomlims[1][1])
handles = [contourlist[j].legend_elements()[0] for j in range(len(contourlist))]
ax.legend([handles[j][0] for j in range(len(contourlist))], lines)

# markerpospix = wcs.all_world2pix(52.2819472, 31.3654766, 0)


# ax.set_title(title)

fig.savefig(plotname, bbox_inches='tight')
