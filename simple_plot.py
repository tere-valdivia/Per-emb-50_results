import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.patches import Ellipse, Rectangle
from astropy.visualization import simple_norm
from NOEMAsetup import *
import matplotlib.cm as cm
import copy
import pyregion
# Simple image plotting

#####################
saveaction = 0
regionplot = 1
# could be integration, velocity, velocity width, whatever you want!
folder = 'SO_55_44/CDconfig/'
filename = folder + 'Per-emb-50_CD_l009l048_uvsub_SO_multi_velocity_python'
title = r'SO($5_{5}-4_{4}$) \\ Velocity field \\ C+D Multi' #
stretch = 'linear'
savefile = folder + 'SO_multi_velfield_'+stretch+'_phasecenter_auxregions.png'
# filetitle = filename+'_plot_integrated'
vmin = 6.9
vmax = 8.8
radiusplot = 6/3600
centerplot = [52.2814509,31.3647748]
regionfile = folder+'spectra_aux_reg2.reg'

# logscale = False
cmap = copy.copy(cm.get_cmap("bwr"))
coloraxisname = r'km s$^{-1}$'
contlevels = [0.007]
scalebarsize = 1000

#####################

data = fits.getdata(filename+'.fits')
header = fits.getheader(filename+'.fits')
wcs = WCS(header).celestial
if regionplot:
    regions = pyregion.open(regionfile)
ra = header['RA']
dec = header['DEC']
freq = header['RESTFREQ']* u.Hz
pixsize = header['CDELT2']

bmaj = header["BMAJ"]/pixsize
bmin = header['BMIN']/pixsize
bpa = header['BPA']
continuum = fits.getdata('SO_55_44/Per-emb-50_C_l009l048_cont.fits')[0]
radiusPB = (pb_noema(freq).to(u.deg)*0.5).value

phasecentpix = wcs.all_world2pix(ra, dec, 0)

markerpospix = wcs.all_world2pix(ra_Per50, dec_Per50, 0)

zoomlims = wcs.all_world2pix([ra-radiusplot, ra+radiusplot],
                             [dec-radiusplot, dec+radiusplot], 0)
# zoomlims = wcs.all_world2pix([centerplot[0]-radiusplot, centerplot[0]+radiusplot],
#                              [centerplot[1]-radiusplot, centerplot[1]+radiusplot], 0)
circle1 = plt.Circle(phasecentpix, radiusPB /
                     pixsize, edgecolor='w', fill=False, linestyle='--')
beam = Ellipse([zoomlims[0][1]+10, zoomlims[1][0]+10], bmaj,
               bmin, -bpa, facecolor='k', edgecolor='k')
beammarquee = Rectangle([zoomlims[0][1], zoomlims[1][0]], 20, 20, facecolor='w')

if regionplot:
    r2 = regions.as_imagecoord(header)
    patch_list, artist_list = r2.get_mpl_patches_texts()

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection=wcs)
norm = simple_norm(data, stretch,min_cut=vmin, max_cut=vmax)

cmap.set_bad('k')
im = ax.imshow(data, cmap=cmap, norm=norm)
fig.colorbar(im, ax=ax, label=coloraxisname)
ax.set_xlabel('RA (ICRS)')
ax.set_ylabel('DEC (ICRS)')
ax.scatter(phasecentpix[0], phasecentpix[1], c='k', s=20,marker='x', label='Phase center')
ax.set_xlim(zoomlims[0][1], zoomlims[0][0])
ax.set_ylim(zoomlims[1][0], zoomlims[1][1])
ax.add_patch(beammarquee)
ax.add_patch(beam)
ax.add_artist(circle1)
if regionplot:
    for p in patch_list:
        ax.add_patch(p)
    for t in artist_list:
        ax.add_artist(t)
scale =((scalebarsize / dist_Per50) * u.arcsec).to(u.deg).value/pixsize
fontprops = fm.FontProperties(size=14)
scalebar = AnchoredSizeBar(ax.transData,
                           scale, '1000 AU', 'lower center',
                           pad=0.1,
                           color='white',
                           frameon=False,
                           size_vertical=0.5,
                           fontproperties=fontprops)
contourc = ax.contour(continuum, levels=contlevels, colors='c', label='215 GHz cont.')
ax.add_artist(scalebar)
# ax.set_title(title)
ax.text(0.45,0.91, title, color='w',horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, size=16)
if saveaction:
    fig.savefig(savefile, bbox_inches='tight')
