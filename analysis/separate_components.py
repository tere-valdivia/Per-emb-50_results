
'''
Steps:

Identify the criteria to separate the streamer from the rest of the components

'''

import numpy as np
import pyspeckit
from astropy.wcs import WCS
from astropy.io import fits
import matplotlib.pyplot as plt

cubefile = '../SO_55_44/CDconfigsmall/Per-emb-50_CD_l009l048_uvsub_SO_multi_small'
fitfile1aicres = cubefile + '_1G_fitparams_aicres.fits'
fitfile2aicres = cubefile + '_2G_fitparams_aicres.fits'
fitfile3aicres = cubefile + '_3G_fitparams_aicres.fits'

# load components
header = fits.getheader(fitfile1aicres)
wcs = WCS(header).celestial
comp_1G = fits.getdata(fitfile1aicres)
comp1_1G = comp_1G[:3]
comp_2G = fits.getdata(fitfile2aicres)
comp1_2G = comp_2G[:3]
comp2_2G = comp_2G[3:6] # remember they have the errors as well
comp_3G = fits.getdata(fitfile3aicres)
comp1_3G = comp_3G[:3]
comp2_3G = comp_3G[3:6]
comp3_3G = comp_3G[6:9]

#lets see where the sigma is less than 0.4
# we also know that it must be blueshifted
map_sigma_05 = np.zeros(np.shape(comp1_1G))*np.nan
np.multiply(comp1_1G[2]<0.4,comp1_1G[1]<7.5)
index_sigma = np.where(np.multiply(comp1_1G[2]<0.5, comp1_1G[1]<7.5))
map_sigma_05[0, index_sigma[0], index_sigma[1]] = comp1_1G[0, index_sigma[0], index_sigma[1]]
map_sigma_05[1, index_sigma[0], index_sigma[1]] = comp1_1G[1, index_sigma[0], index_sigma[1]]
map_sigma_05[2, index_sigma[0], index_sigma[1]] = comp1_1G[2, index_sigma[0], index_sigma[1]]

index_sigma2 = np.where(np.multiply(comp1_2G[2]<0.5, comp1_2G[1]<7.5))
map_sigma_05[0, index_sigma2[0], index_sigma2[1]] = comp1_2G[0, index_sigma2[0], index_sigma2[1]]
map_sigma_05[1, index_sigma2[0], index_sigma2[1]] = comp1_2G[1, index_sigma2[0], index_sigma2[1]]
map_sigma_05[2, index_sigma2[0], index_sigma2[1]] = comp1_2G[2, index_sigma2[0], index_sigma2[1]]

index_sigma3 = np.where(np.multiply(comp2_3G[2]<0.5, comp2_3G[1]<7.5))
map_sigma_05[0, index_sigma3[0], index_sigma3[1]] = comp2_3G[0, index_sigma3[0], index_sigma3[1]]
map_sigma_05[1, index_sigma3[0], index_sigma3[1]] = comp2_3G[1, index_sigma3[0], index_sigma3[1]]
map_sigma_05[2, index_sigma3[0], index_sigma3[1]] = comp2_3G[2, index_sigma3[0], index_sigma3[1]]
cmap = plt.cm.RdYlBu_r
cmap.set_bad(np.array([1,1,1])*0.95)
fig = plt.figure(figsize=(4,12))
ax = fig.add_subplot(311, projection=wcs)
im = ax.imshow(map_sigma_05[0])
fig.colorbar(im, ax=ax)
ax2 = fig.add_subplot(312, projection=wcs)
im2 = ax2.imshow(map_sigma_05[1], vmin=6.5,vmax=7.5, cmap=cmap)
fig.colorbar(im2, ax=ax2)
ax3 = fig.add_subplot(313, projection=wcs)
im3 = ax3.imshow(map_sigma_05[2])
fig.colorbar(im3, ax=ax3)
