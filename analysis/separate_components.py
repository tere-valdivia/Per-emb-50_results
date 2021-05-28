
'''
We have fitted 3 gaussian components in the region we call fitcube 2g.
We have fitted only one gaussian component for the whole map (called small)
To get the streamer component and kink, we select the parts of the map which
have sigma<0.8 and velo<7.5 km/s
Then we insert the central part to the whole map

'''

import numpy as np
import pyspeckit
from astropy.wcs import WCS
from astropy.io import fits
import matplotlib.pyplot as plt
import aplpy
from spectral_cube import SpectralCube
from astropy.modeling.functional_models import Gaussian1D


cubefile = '../SO_55_44/CDconfigsmall/gaussian_fit_123G_fitcube2g/Per-emb-50_CD_l009l048_uvsub_SO_multi_small'
fitfile1aicres = cubefile + '_1G_fitparams_aicres.fits'
fitfile2aicres = cubefile + '_2G_fitparams_aicres.fits'
fitfile3aicres = cubefile + '_3G_fitparams_aicres.fits'
cube = SpectralCube.read(cubefile+'_fitcube2g_K.fits')
streamersavename = '../SO_55_44/CDconfigsmall/streamer_component_123G_corrected_0.8kms.pdf'
fitstreamfile = cubefile+'_gaussian_streamer_model.fits'
cubetotalfile = '../SO_55_44/CDconfigsmall/Per-emb-50_CD_l009l048_uvsub_SO_multi_small'
fittotalfile = cubetotalfile + '_gaussian_streamer_kink_model.fits'
dispersionmax = 0.8


if os.path.exists(fitstreamfile):
    map_sigma = fits.getdata(fitstreamfile)
    header = fits.getheader(fitstreamfile)
else:
    # Do the selection of the streamer component from the gaussians
    # load components
    header = fits.getheader(fitfile1aicres)
    wcs = WCS(header).celestial
    comp_1G = fits.getdata(fitfile1aicres)
    comp1_1G = comp_1G[:3] # remember they have the errors as well
    ecomp1_1G = comp_1G[3:]
    comp_2G = fits.getdata(fitfile2aicres)
    comp1_2G = comp_2G[:3]
    comp2_2G = comp_2G[3:6]
    ecomp1_2G = comp_2G[6:9]
    ecomp2_2G = comp_2G[9:]
    comp_3G = fits.getdata(fitfile3aicres)
    comp1_3G = comp_3G[:3]
    comp2_3G = comp_3G[3:6]
    comp3_3G = comp_3G[6:9]
    ecomp1_3G = comp_3G[9:12]
    ecomp2_3G = comp_3G[12:15]
    ecomp3_3G = comp_3G[15:]

    map_sigma = np.zeros(np.shape(comp_1G))*np.nan # We save errors as well
    xx, yy = np.meshgrid(range(np.shape(comp1_1G)[2]),range(np.shape(comp1_1G)[1]))

    # components that I think  correspond to the streamer
    comp1G_streamer = comp1_1G
    comp2G_streamer = comp1_2G
    comp3G_streamer = comp2_3G
    ecomp1G_streamer = ecomp1_1G
    ecomp2G_streamer = ecomp1_2G
    ecomp3G_streamer = ecomp2_3G

    index_sigma = np.where(np.multiply(comp1G_streamer[2]<dispersionmax, comp1G_streamer[1]<7.5))
    map_sigma[0, index_sigma[0], index_sigma[1]] = comp1G_streamer[0, index_sigma[0], index_sigma[1]]
    map_sigma[1, index_sigma[0], index_sigma[1]] = comp1G_streamer[1, index_sigma[0], index_sigma[1]]
    map_sigma[2, index_sigma[0], index_sigma[1]] = comp1G_streamer[2, index_sigma[0], index_sigma[1]]
    map_sigma[3, index_sigma[0], index_sigma[1]] = ecomp1G_streamer[0, index_sigma[0], index_sigma[1]]
    map_sigma[4, index_sigma[0], index_sigma[1]] = ecomp1G_streamer[1, index_sigma[0], index_sigma[1]]
    map_sigma[5, index_sigma[0], index_sigma[1]] = ecomp1G_streamer[2, index_sigma[0], index_sigma[1]]

    index_sigma2 = np.where(np.multiply(comp2G_streamer[2]<dispersionmax, comp2G_streamer[1]<7.5))
    map_sigma[0, index_sigma2[0], index_sigma2[1]] = comp2G_streamer[0, index_sigma2[0], index_sigma2[1]]
    map_sigma[1, index_sigma2[0], index_sigma2[1]] = comp2G_streamer[1, index_sigma2[0], index_sigma2[1]]
    map_sigma[2, index_sigma2[0], index_sigma2[1]] = comp2G_streamer[2, index_sigma2[0], index_sigma2[1]]
    map_sigma[3, index_sigma2[0], index_sigma2[1]] = ecomp2G_streamer[0, index_sigma2[0], index_sigma2[1]]
    map_sigma[4, index_sigma2[0], index_sigma2[1]] = ecomp2G_streamer[1, index_sigma2[0], index_sigma2[1]]
    map_sigma[5, index_sigma2[0], index_sigma2[1]] = ecomp2G_streamer[2, index_sigma2[0], index_sigma2[1]]

    # We change some pixels by hand
    index_sigma2_corr = (np.array([29, 16]),np.array([33,37]))
    map_sigma[0, index_sigma2_corr[0], index_sigma2_corr[1]] = comp2G_streamer[0, index_sigma2_corr[0], index_sigma2_corr[1]]
    map_sigma[1, index_sigma2_corr[0], index_sigma2_corr[1]] = comp2G_streamer[1, index_sigma2_corr[0], index_sigma2_corr[1]]
    map_sigma[2, index_sigma2_corr[0], index_sigma2_corr[1]] = comp2G_streamer[2, index_sigma2_corr[0], index_sigma2_corr[1]]
    map_sigma[3, index_sigma2_corr[0], index_sigma2_corr[1]] = ecomp2G_streamer[0, index_sigma2_corr[0], index_sigma2_corr[1]]
    map_sigma[4, index_sigma2_corr[0], index_sigma2_corr[1]] = ecomp2G_streamer[1, index_sigma2_corr[0], index_sigma2_corr[1]]
    map_sigma[5, index_sigma2_corr[0], index_sigma2_corr[1]] = ecomp2G_streamer[2, index_sigma2_corr[0], index_sigma2_corr[1]]

    index_sigma3 = np.where(np.multiply(comp3G_streamer[2]<dispersionmax, comp3G_streamer[1]<7.5))
    map_sigma[0, index_sigma3[0], index_sigma3[1]] = comp3G_streamer[0, index_sigma3[0], index_sigma3[1]]
    map_sigma[1, index_sigma3[0], index_sigma3[1]] = comp3G_streamer[1, index_sigma3[0], index_sigma3[1]]
    map_sigma[2, index_sigma3[0], index_sigma3[1]] = comp3G_streamer[2, index_sigma3[0], index_sigma3[1]]
    map_sigma[3, index_sigma3[0], index_sigma3[1]] = ecomp3G_streamer[0, index_sigma3[0], index_sigma3[1]]
    map_sigma[4, index_sigma3[0], index_sigma3[1]] = ecomp3G_streamer[1, index_sigma3[0], index_sigma3[1]]
    map_sigma[5, index_sigma3[0], index_sigma3[1]] = ecomp3G_streamer[2, index_sigma3[0], index_sigma3[1]]

    # We know that for pixels 34,25 and 34,26 (x,y) it is best to use the 1st component of the 3G

    index_sigma3_corr = (np.array([25,26]),np.array([34, 34]))
    map_sigma[0, index_sigma3_corr[0], index_sigma3_corr[1]] = comp1_3G[0, index_sigma3_corr[0], index_sigma3_corr[1]]
    map_sigma[1, index_sigma3_corr[0], index_sigma3_corr[1]] = comp1_3G[1, index_sigma3_corr[0], index_sigma3_corr[1]]
    map_sigma[2, index_sigma3_corr[0], index_sigma3_corr[1]] = comp1_3G[2, index_sigma3_corr[0], index_sigma3_corr[1]]
    map_sigma[3, index_sigma3_corr[0], index_sigma3_corr[1]] = ecomp1_3G[0, index_sigma3_corr[0], index_sigma3_corr[1]]
    map_sigma[4, index_sigma3_corr[0], index_sigma3_corr[1]] = ecomp1_3G[1, index_sigma3_corr[0], index_sigma3_corr[1]]
    map_sigma[5, index_sigma3_corr[0], index_sigma3_corr[1]] = ecomp1_3G[2, index_sigma3_corr[0], index_sigma3_corr[1]]

    filter = np.where(yy>30,1,0)

    map_sigma[np.where(np.repeat([filter], 6, axis=0))]  = np.nan

labelx = 'Right Ascension (J2000)'
labely = 'Declination (J2000)'
cmap = plt.cm.RdYlBu_r
cmap_inferno = plt.cm.inferno
cmap.set_bad(np.array([1,1,1])*0.85)
cmap_inferno.set_bad(np.array([1,1,1])*0.85)

#save in plot
fig = plt.figure(figsize=(4,12))
ax = fig.add_subplot(311, projection=wcs)
im = ax.imshow(map_sigma[0],cmap=cmap_inferno, vmin=0, vmax=5)
fig.colorbar(im, ax=ax, label='Peak (K)')
ax.set_xlabel(labelx)
ax.set_ylabel(labely)
ax2 = fig.add_subplot(312, projection=wcs)
im2 = ax2.imshow(map_sigma[1],vmin=6.5,vmax=7.5, cmap=cmap)
fig.colorbar(im2, ax=ax2, label='Velocity (km/s)')
ax2.set_xlabel(labelx)
ax2.set_ylabel(labely)
ax3 = fig.add_subplot(313, projection=wcs)
im3 = ax3.imshow(map_sigma[2], cmap=cmap_inferno, origin='lower', vmin=0, vmax=0.8)
fig.colorbar(im3, ax=ax3, label='Velocity dispersion (km/s)')
ax3.set_xlabel(labelx)
ax3.set_ylabel(labely)

# fig.savefig(streamersavename,dpi=300,bbox_inches='tight')

#Save in fits
# hdu = fits.PrimaryHDU(data=map_sigma, header=header)
# hdu.writeto(fitstreamfile)

# im3 = ax3.imshow(yy,cmap=cmap_inferno,origin='lower')
# fig.colorbar(im3, ax=ax3)

# We can see the spectra of the 3 component only to see if we are choosing
# the correct component for each pixel
# indexes = np.where(~np.isnan(map_sigma[0]))
# velo = cube.spectral_axis
# for y,x in zip(indexes[0],indexes[1]):
#     g_stream = Gaussian1D(map_sigma[0,y,x], map_sigma[1,y,x], map_sigma[2,y,x])
#     spectrum = cube.unmasked_data[:,y,x]
#     fig_sp = plt.figure(figsize=(6,6))
#     ax_sp = fig_sp.add_subplot(111)
#     ax_sp.plot(velo,spectrum)
#     ax_sp.plot(velo, g_stream(velo.value))
#     ax_sp.set_title(str(x)+', '+str(y))
#     fig_sp.savefig('../SO_55_44/CDconfigsmall/pixel_2g_analysis/comp1_2g_'+str(x)+'_'+str(y)+'.png')

# ax_sp2.plot(velo, g_stream(velo.value))
# Now we get to see what is left

# comp1G_streamer[:, index_sigma[0], index_sigma[1]] = np.nan
# comp2G_streamer[:, index_sigma2[0], index_sigma2[1]] = np.nan
# comp3G_streamer[:, index_sigma3[0], index_sigma3[1]] = np.nan
#
# fig = plt.figure(figsize=(4,12))
# ax = fig.add_subplot(311, projection=wcs)
# im = ax.imshow(comp1G_streamer[0],cmap=cmap_inferno)
# fig.colorbar(im, ax=ax, label='Peak (K)')
# ax.set_xlabel(labelx)
# ax.set_ylabel(labely)
# ax2 = fig.add_subplot(312, projection=wcs)
# im2 = ax2.imshow(comp1G_streamer[1],vmin=6,vmax=9, cmap=cmap)
# fig.colorbar(im2, ax=ax2, label='Velocity (km/s)')
# ax2.set_xlabel(labelx)
# ax2.set_ylabel(labely)
# ax3 = fig.add_subplot(313, projection=wcs)
# im3 = ax3.imshow(comp1G_streamer[2], cmap=cmap_inferno, origin='lower')
# fig.colorbar(im3, ax=ax3, label='Velocity dispersion (km/s)')
# ax3.set_xlabel(labelx)
# ax3.set_ylabel(labely)
