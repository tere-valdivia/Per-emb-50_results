import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from spectral_cube import SpectralCube


savequestion = True
namecomparison = 'SO2_intensities.eps'
label1 = 'SO2_4_22_3_13'
label2 = 'SO2_11_1_11_10_0_10'
label3 = 'SO2_12_39_12_210'

folder = ''#'H2CO/'
fitsfile = folder+'SO2_4_22_3_13/Per-emb-50_C_l042l081_uvsub_SO2_multi'
fitsfile2 = folder+'SO2_11_1_11_10_0_10/Per-emb-50_C_l031l070_uvsub_SO2_multi'
fitsfile3 = folder+'SO2_12_39_12_210/Per-emb-50_C_l046l085_uvsub_SO2_multi'
fitsfilelist = [fitsfile, fitsfile2, fitsfile3]
labellist = [label1, label2, label3]
colorlist = ['green', 'blue', 'red']

velstart = 2
velend = 13


phasecent = np.array([52.28236666667, 31.36586888889])
# pos1 = phasecent  # at phasecent
pos1 = np.array([52.2803453, 31.3623526])
# pos1 = np.array([52.2802127, 31.3622965])#


def degtohoursRA(deg):
    hours = deg / 360 * 24
    hh = np.trunc(hours).astype('int64')

    minutes = ((hours-hh) % 60) * 60
    mm = np.trunc(minutes).astype('int64')
    seconds = ((minutes-mm) % 60) * 60
    ss = np.round(seconds, 2)
    return '{0:02d}:{1:02d}:{2:0>5}'.format(hh, mm, ss)


def degtohoursDEC(deg):
    dd = np.trunc(deg).astype('int64')
    minutes = ((np.abs(deg)-np.abs(dd)) % 60) * 60
    mm = np.trunc(minutes).astype('int64')
    seconds = ((minutes-mm) % 60) * 60
    ss = np.round(seconds, 2)
    return '{0:02d}:{1:02d}:{2:0>5}'.format(dd, mm, ss)


def plotSpectra(ax, vel, spec, label=None, xlims=None, ylims=None, color=None):
    ax.plot(vel, spec, drawstyle='steps-mid', label=label, color=color)
    ax.set_xlabel('Velocity (km/s)')
    ax.set_ylabel('Intensity (Jy/beam)')

    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    return ax


def plotSpectras(fig, position, fitsfiles, labels, colorlist, phasecenter, subplots=111, xlims=None, ylims=None, saving=False, specname=''):
    deltas = phasecent-position
    rah = degtohoursRA(pos1[0])
    dech = degtohoursDEC(pos1[1])
    ax = fig.add_subplot(subplots)
    for fitsfile, label, color in zip(fitsfiles, labels, colorlist):
        cube = SpectralCube.read(fitsfile+'.fits')
        cube = cube.with_spectral_unit(u.km/u.s)
        cube = cube.spectral_slab(velstart*u.km/u.s, velend*u.km/u.s)
        # cube = fits.getdata(fitsfile+'.fits')
        cubehead = fits.getheader(fitsfile+'.fits')
        cubewcs = WCS(cubehead)
        ra, dec = cubewcs.celestial.all_world2pix(position[0], position[1], 0)
        velocities = cube.spectral_axis
        spectra = cube[:, int(dec), int(ra)]
        ax = plotSpectra(ax, velocities, spectra, label=label, xlims=xlims, ylims=ylims, color=color)

    ax.legend(loc=0)
    ax.set_title(r'Spectra at $\Delta$ ' +
                 str(np.round(deltas[0]*3600, 2))+', '+str(np.round(-deltas[1]*3600, 2))+' ('+rah+', '+dech+')')
    if saving:
        fig.savefig('delta_'+str(np.round(deltas[0]*3600, 2))+'_' + str(
            np.round(-deltas[1]*3600, 2))+'_'+specname+'.png', bbox_inches='tight', transparent=False)


plt.clf()
fig = plt.figure(figsize=(8, 6))
plotSpectras(fig, pos1, fitsfilelist, labellist, colorlist, phasecent,
             xlims=[velstart, velend], ylims=[-0.05, 0.2], saving=savequestion, specname=namecomparison)
