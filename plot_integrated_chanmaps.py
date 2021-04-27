import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from spectral_cube import SpectralCube
from matplotlib.patches import Ellipse, Rectangle
from astropy.visualization import simple_norm
import matplotlib.font_manager as fm
from NOEMAsetup import *
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import astropy.units as u

#############
# chauvenet: factor de correlacion, no tener la delta v escrita pero sacarlo del header
# This should be the only things to modify
saveaction = 1
folder = 'SO_55_44/CDconfigsmall/'
cubename = folder + 'Per-emb-50_CD_l009l048_uvsub_SO_multi_small_fitcube2g'
nrows = 4
ncols = 5
velinit = 5
velend = 11
stretch = 'asinh'
scalebarsize = 300 # au
plotname = cubename + '_integratedchanmaps_'+str(velinit)+'to'+str(velend)+'_'+str(ncols*nrows)+'cells_'+stretch+'.pdf'
deltav = (velend-velinit)/(nrows*ncols)
radiusplot = 7/3600
# logscale = False
cmap = 'inferno'
coloraxisname = 'K km/s'
vmin = 0
vmax = 8
textsizept = 12
figsize = (10,8)

##############

cube = SpectralCube.read(cubename+'.fits')
cube = cube.with_spectral_unit(u.km/u.s)
header = cube.hdu.header
pixsize = header['CDELT2']
wcs = WCS(header).celestial
ra = header['RA']
dec = header['DEC']
bmaj = header["BMAJ"]/pixsize
bmin = header['BMIN']/pixsize
bpa = header['BPA']
radiusPB = (pb_noema(header['RESTFREQ']* u.Hz)).to(u.deg).value/ pixsize*0.5
scale = ((scalebarsize/293)*u.arcsec).to(u.deg).value / pixsize


phasecentpix = wcs.all_world2pix(ra, dec, 0)

velrange = np.linspace(velinit, velend, int((velend-velinit)/deltav+1))

fig, axeslist = plt.subplots(nrows, ncols, sharex='col', sharey='row', subplot_kw={'projection':wcs}, figsize=figsize)
# zoomlims = wcs.all_world2pix([ra-radiusplot, ra+radiusplot],
#                              [dec-radiusplot, dec+radiusplot], 0)


fontprops = fm.FontProperties(size=14)

i = 0
for row in range(nrows):
    for col in range(ncols):
        ax = axeslist[row, col]
        velstarti = velrange[i]
        velendi = velrange[i+1]
        moment0 = cube.spectral_slab(velstarti*u.km/u.s, velendi*u.km/u.s).moment(order=0).value
        norm = simple_norm(moment0, stretch)
        im = ax.imshow(moment0, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
        circle1 = plt.Circle(phasecentpix, radiusPB, edgecolor='w', fill=False, linestyle='--')
        # beam = Ellipse([zoomlims[0][1]+15, zoomlims[1][0]+15], bmaj,
        #                bmin, -bpa, facecolor='k', edgecolor='k')
        beam = Ellipse([5, 5], bmaj,
                       bmin, bpa-90, facecolor='k', edgecolor='k')
        # beammarquee = Rectangle([zoomlims[0][1], zoomlims[1][0]], 30, 30, facecolor='w')
        beammarquee = Rectangle([0, 0], 10, 10, facecolor='w')
        # ax.set_xlim(zoomlims[0][1], zoomlims[0][0])
        # ax.set_ylim(zoomlims[1][0], zoomlims[1][1])
        markerpospix = wcs.all_world2pix(ra_Per50, dec_Per50, 0)
        ax.scatter(phasecentpix[0], phasecentpix[1], c='c', s=20, marker='x', label='Phase center')
        ax.text(0.5, 0.9, str(round(velstarti, 2))+' to '+str(round(velendi, 2))+' km/s', fontdict=dict(color='w', size=textsizept), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        if i==0:
            ax.add_patch(beammarquee)
            ax.add_patch(beam)
            fontprops = fm.FontProperties(size=12)
            scalebar = AnchoredSizeBar(ax.transData,
                                       scale, '300 AU', 'lower right',
                                       pad=0.1,
                                       color='white',
                                       frameon=False,
                                       size_vertical=1,
                                       fontproperties=fontprops)
            ax.add_artist(scalebar)
        ax.add_artist(circle1)

        i+=1
        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.set_major_formatter('dd:mm:ss')
        lon.set_axislabel('RA (J2000)')
        lat.set_axislabel('DEC (J2000)')
        if row!=nrows-1 or col!=0:
            lon.set_ticks_visible(False)
            lon.set_ticklabel_visible(False)
            lat.set_ticks_visible(False)
            lat.set_ticklabel_visible(False)

plt.subplots_adjust(wspace=0, hspace=0)
cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax, label=coloraxisname)
if saveaction:
    fig.savefig(plotname, bbox_inches='tight')
# # fig = plt.figure(figsize=(10, 6))
#
# for i in range(len(velrange)-1):
#     subcube = cube.spectral_slab(velrange[i]*u.km/u.s, velrange[i+1]*u.km/u.s)
#     moment0 = subcube.moment(order=0).value
#     ax = fig.add_subplot(nrows, ncols, i+1, projection=wcs)
#     ax.imshow(moment0, origin='lower')
