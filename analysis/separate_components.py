
'''
For SO:
We have fitted 3 gaussian components in the region we call fitcube 2g.
We have fitted only one gaussian component for the whole map (called small)

'''
import os
import numpy as np
import pyspeckit
from astropy.wcs import WCS
from astropy.io import fits
import matplotlib.pyplot as plt
import aplpy
from spectral_cube import SpectralCube
from astropy.modeling.functional_models import Gaussian1D
import copy


cubefile = '../SO_55_44/CDconfigsmall/gaussian_fit_123G_fitcube2g/Per-emb-50_CD_l009l048_uvsub_SO_multi_small'
fitfile1aicres = cubefile + '_1G_fitparams_aicres.fits'
fitfile2aicres = cubefile + '_2G_fitparams_aicres.fits'
fitfile3aicres = cubefile + '_3G_fitparams_aicres.fits'
# cube = SpectralCube.read(cubefile+'_fitcube2g_K.fits')

streamersavename = '../SO_55_44/CDconfigsmall/streamer_component_123G_corrected_0.8kms_samescale.pdf'
fitstreamfile = cubefile+'_gaussian_streamer_model.fits'
fitstreamfile_components = cubefile+'_gaussian_streamer_model_components.fits'
velmaxstream = 7.2
maxypixstream = 34
# cubetotalfile = '../SO_55_44/CDconfigsmall/Per-emb-50_CD_l009l048_uvsub_SO_multi_small'
# fittotalfile = cubetotalfile + '_gaussian_streamer_kink_model.fits'

rotsavename = '../SO_55_44/CDconfigsmall/rotation_component_123G.pdf'
fitrotfile = cubefile+'_gaussian_rotation_model.fits'
fitrotfile_components = cubefile+'_gaussian_rotation_model_components.fits'

infsavename = '../SO_55_44/CDconfigsmall/rest_component_123G.pdf'
fitinffile = cubefile+'_gaussian_infall_model.fits'
fitinffile_components = cubefile+'_gaussian_infall_model_components.fits'
velinfmin = 8.1

wingsavename = '../SO_55_44/CDconfigsmall/rest_component_123G.pdf'
fitwingsfile = cubefile+'_gaussian_wings_model.fits'
fitwingsfile_components = cubefile+'_gaussian_wings_model_components.fits'
dispersionminwing = 2.0

# load components
header = fits.getheader(fitfile1aicres)
wcs = WCS(header).celestial
comp_1G = fits.getdata(fitfile1aicres)
comp1_1G = comp_1G[:3]
ecomp1_1G = comp_1G[3:]  # remember they have the errors as well
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
components = np.array([comp1_1G, comp1_2G, comp2_2G, comp1_3G, comp2_3G, comp3_3G])
comperrors = np.array([ecomp1_1G, ecomp1_2G, ecomp2_2G, ecomp1_3G, ecomp2_3G, ecomp3_3G])

xx, yy = np.meshgrid(range(np.shape(comp1_1G)[2]),range(np.shape(comp1_1G)[1]))

map_stream = np.zeros(np.shape(components)) * np.nan
emap_stream = np.zeros(np.shape(components)) * np.nan  # We save errors as well
map_rot = np.zeros(np.shape(components)) * np.nan
emap_rot = np.zeros(np.shape(components)) * np.nan
map_wings = np.zeros(np.shape(components)) * np.nan
emap_wings = np.zeros(np.shape(components)) * np.nan
map_inf = np.zeros(np.shape(components)) * np.nan
emap_inf = np.zeros(np.shape(components)) * np.nan

flagmap_wings = np.zeros(np.shape(comp_1G[0]))
flagmap_inf = np.zeros(np.shape(comp_1G[0]))
flagmap_stream = np.zeros(np.shape(comp_1G[0]))
flagmap_rot = np.zeros(np.shape(comp_1G[0]))

for ind in range(len(components)):
    # we evaluate components with large dispersion, because those are
    # surely wings
    criteria_wings = components[ind][2] > dispersionminwing
    index_wings = np.where(criteria_wings)
    map_wings[ind, 0, index_wings[0], index_wings[1]] = components[ind][0, index_wings[0], index_wings[1]]
    map_wings[ind, 1, index_wings[0], index_wings[1]] = components[ind][1, index_wings[0], index_wings[1]]
    map_wings[ind, 2, index_wings[0], index_wings[1]] = components[ind][2, index_wings[0], index_wings[1]]
    emap_wings[ind, 0, index_wings[0], index_wings[1]] = comperrors[ind][0, index_wings[0], index_wings[1]]
    emap_wings[ind, 1, index_wings[0], index_wings[1]] = comperrors[ind][1, index_wings[0], index_wings[1]]
    emap_wings[ind, 2, index_wings[0], index_wings[1]] = comperrors[ind][2, index_wings[0], index_wings[1]]
    flagmap_wings[index_wings] = ind + 1

    # now we select the prevalent high velocity component,adding the opposite condition of the previous
    criteria_inf = np.multiply(components[ind][1] > velinfmin, components[ind][2] < dispersionminwing)
    index_rest = np.where(criteria_inf)
    map_inf[ind, 0, index_rest[0], index_rest[1]] = components[ind][0, index_rest[0], index_rest[1]]
    map_inf[ind, 1, index_rest[0], index_rest[1]] = components[ind][1, index_rest[0], index_rest[1]]
    map_inf[ind, 2, index_rest[0], index_rest[1]] = components[ind][2, index_rest[0], index_rest[1]]
    emap_inf[ind, 0, index_rest[0], index_rest[1]] = comperrors[ind][0, index_rest[0], index_rest[1]]
    emap_inf[ind, 1, index_rest[0], index_rest[1]] = comperrors[ind][1, index_rest[0], index_rest[1]]
    emap_inf[ind, 2, index_rest[0], index_rest[1]] = comperrors[ind][2, index_rest[0], index_rest[1]]
    flagmap_inf[index_rest] = ind + 1

    # now we select the streamer component component, adding the opposite condition of the previous
    # here we have 2 criteria: where the velocity is lower than velmax or where the velocity is less than velmax2 and towards the south
    criteria_stream = np.multiply(components[ind][1] < velmaxstream, np.multiply(yy<maxypixstream, components[ind][2] < dispersionminwing), dtype=int)
    index_stream = np.where(criteria_stream)
    map_stream[ind, 0, index_stream[0], index_stream[1]] = components[ind][0, index_stream[0], index_stream[1]]
    map_stream[ind, 1, index_stream[0], index_stream[1]] = components[ind][1, index_stream[0], index_stream[1]]
    map_stream[ind, 2, index_stream[0], index_stream[1]] = components[ind][2, index_stream[0], index_stream[1]]
    emap_stream[ind, 0, index_stream[0], index_stream[1]] = comperrors[ind][0, index_stream[0], index_stream[1]]
    emap_stream[ind, 1, index_stream[0], index_stream[1]] = comperrors[ind][1, index_stream[0], index_stream[1]]
    emap_stream[ind, 2, index_stream[0], index_stream[1]] = comperrors[ind][2, index_stream[0], index_stream[1]]
    flagmap_stream[index_stream] = ind + 1

    #The rest that does not follow the previous criteria should be rotation
    #  heavy emphasis on the 'should'
    criteria_rot = np.sum([~criteria_wings, ~criteria_inf, ~criteria_stream], axis=0)
    index_rot = np.where(np.multiply(criteria_rot, ~np.isnan(components[ind][1])))
    map_rot[ind, 0, index_rot[0], index_rot[1]] = components[ind][0, index_rot[0], index_rot[1]]
    map_rot[ind, 1, index_rot[0], index_rot[1]] = components[ind][1, index_rot[0], index_rot[1]]
    map_rot[ind, 2, index_rot[0], index_rot[1]] = components[ind][2, index_rot[0], index_rot[1]]
    emap_rot[ind, 0, index_rot[0], index_rot[1]] = comperrors[ind][0, index_rot[0], index_rot[1]]
    emap_rot[ind, 1, index_rot[0], index_rot[1]] = comperrors[ind][1, index_rot[0], index_rot[1]]
    emap_rot[ind, 2, index_rot[0], index_rot[1]] = comperrors[ind][2, index_rot[0], index_rot[1]]
    flagmap_rot[index_rot] = ind + 1


map_wings_total = np.zeros(np.shape(comp1_1G)) * np.nan  # 3 pages
emap_wings_total = np.zeros(np.shape(comp1_1G)) * np.nan  # 3 pages
for i in range(len(map_wings)):
    index_fill = np.where(~np.isnan(map_wings[i]))
    map_wings_total[index_fill] = map_wings[i][index_fill]
    emap_wings_total[index_fill] = emap_wings[i][index_fill]
map_wings = np.concatenate([map_wings, emap_wings])
map_wings_total = np.concatenate([map_wings_total,emap_wings_total])

map_inf_total = np.zeros(np.shape(comp1_1G)) * np.nan  # 3 pages
emap_inf_total = np.zeros(np.shape(comp1_1G)) * np.nan  # 3 pages
for i in range(len(map_inf)):
    index_fill = np.where(~np.isnan(map_inf[i]))
    map_inf_total[index_fill] = map_inf[i][index_fill]
    emap_inf_total[index_fill] = emap_inf[i][index_fill]
map_inf = np.concatenate([map_inf, emap_inf])
map_inf_total = np.concatenate([map_inf_total,emap_inf_total])

map_stream_total = np.zeros(np.shape(comp1_1G)) * np.nan  # 3 pages
emap_stream_total = np.zeros(np.shape(comp1_1G)) * np.nan  # 3 pages
for i in range(len(map_stream)):
    index_fill = np.where(~np.isnan(map_stream[i]))
    map_stream_total[index_fill] = map_stream[i][index_fill]
    emap_stream_total[index_fill] = emap_stream[i][index_fill]
map_stream = np.concatenate([map_stream, emap_stream])
map_stream_total = np.concatenate([map_stream_total, emap_stream_total])

map_rot_total = np.zeros(np.shape(comp1_1G)) * np.nan  # 3 pages
emap_rot_total = np.zeros(np.shape(comp1_1G)) * np.nan  # 3 pages
for i in range(len(map_rot)):
    index_fill = np.where(~np.isnan(map_rot[i]))
    map_rot_total[index_fill] = map_rot[i][index_fill]
    emap_rot_total[index_fill] = emap_rot[i][index_fill]
map_rot = np.concatenate([map_rot, emap_rot])
map_rot_total = np.concatenate([map_rot_total, emap_rot_total])

# always check if there are no repeated components
#save wings
hdu = fits.PrimaryHDU(data=map_wings, header=header)
hdu.writeto(fitwingsfile_components)
hdu = fits.PrimaryHDU(data=np.concatenate([map_wings_total, [flagmap_wings]]), header=header)
hdu.writeto(fitwingsfile)
#save infall
hdu = fits.PrimaryHDU(data=map_inf, header=header)
hdu.writeto(fitinffile_components)
hdu = fits.PrimaryHDU(data=np.concatenate([map_inf_total, [flagmap_inf]]), header=header)
hdu.writeto(fitinffile)
# save streamer
hdu = fits.PrimaryHDU(data=map_stream, header=header)
hdu.writeto(fitstreamfile_components)
hdu = fits.PrimaryHDU(data=np.concatenate([map_stream_total, [flagmap_stream]]), header=header)
hdu.writeto(fitstreamfile)
# save rotation
hdu = fits.PrimaryHDU(data=map_rot, header=header)
hdu.writeto(fitrotfile_components)
hdu = fits.PrimaryHDU(data=np.concatenate([map_rot_total, [flagmap_rot]]), header=header)
hdu.writeto(fitrotfile)

# labelx = 'Right Ascension (J2000)'
# labely = 'Declination (J2000)'
# cmap = copy.copy(plt.cm.RdYlBu_r)
# cmap_inferno = copy.copy(plt.cm.inferno)
# cmap.set_bad(np.array([1,1,1])*0.85)
# cmap_inferno.set_bad(np.array([1,1,1])*0.85)
#
# #save in plot
# fig = plt.figure(figsize=(4,12))
# ax = fig.add_subplot(311, projection=wcs)
# im = ax.imshow(map_inf[0],cmap=cmap_inferno, vmin=0, vmax=6)
# fig.colorbar(im, ax=ax, label='Peak (K)')
# ax.set_xlabel(labelx)
# ax.set_ylabel(labely)
# ax2 = fig.add_subplot(312, projection=wcs)
# im2 = ax2.imshow(map_inf[1],vmin=5.5,vmax=9.5, cmap=cmap)
# fig.colorbar(im2, ax=ax2, label='Velocity (km/s)')
# ax2.set_xlabel(labelx)
# ax2.set_ylabel(labely)
# ax3 = fig.add_subplot(313, projection=wcs)
# im3 = ax3.imshow(map_inf[2], cmap=cmap_inferno, origin='lower', vmin=0, vmax=5)
# fig.colorbar(im3, ax=ax3, label='Velocity dispersion (km/s)')
# ax3.set_xlabel(labelx)
# ax3.set_ylabel(labely)
# fig.savefig(streamersavename,dpi=300,bbox_inches='tight')
