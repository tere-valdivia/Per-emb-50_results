import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from spectral_cube import SpectralCube
import regions

# Use only with regions!

# comparison = True
saveaction = 1
kelvin = 0
namecomparison = '13CO'
label1 = r'$^{13}$CO($2-1$)'
label2 = None

folder = '13CO/CDconfig/'
fitsfile = folder+'Per-emb-50_CD_l027l066_uvsub_13CO_multi'
fitsfile2 = None

regfile = folder+'spectra_regs.reg'
fitsfilelist = [fitsfile]
labellist = [label1]

velstart = -1
velend = 15

intmin = -0.05
intmax = 0.55 # Jy.beam
# intmin = -5
# intmax = 30 # K

header = fits.getheader(fitsfile+'.fits')
phasecent = np.array([header['ra'], header['dec']])
# pos1 = phasecent  # at phasecent
# pos1 = np.array([52.2819472, 31.3654766])
# pos1 = np.array([52.2802127, 31.3622965])#
regionload = regions.read_ds9(regfile)
positions = [[regionload[i].center.ra.value, regionload[i].center.dec.value]
             for i in range(len(regionload))]
regionlabels = [regionload[i].meta['label'] for i in range(len(regionload))]


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


def plotSpectra(ax, vel, spec, freq=None, beamarea=None, label=None, xlims=None, ylims=None):
    if kelvin:
        spec = spec.to(u.K, u.brightness_temperature(freq, beam_area=beamarea))
    ax.plot(vel, spec, drawstyle='steps-mid', label=label, color='k')
    ax.set_xlabel('Velocity (km/s)')
    if kelvin:
        ax.set_ylabel('Intensity (K)')
    else:
        ax.set_ylabel('Intensity (Jy/beam)')

    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    return ax


def plotSpectras(fig, position, fitsfiles, labels, regionlabel, phasecenter, subplots=111, xlims=None, ylims=None, saving=False, specname=''):
    deltas = phasecent-position
    ax = fig.add_subplot(subplots)
    for fitsfile, label in zip(fitsfiles, labels):
        if fitsfile is None:
            continue
        cube = SpectralCube.read(fitsfile+'.fits')
        cube = cube.with_spectral_unit(u.km/u.s)
        cube = cube.spectral_slab(velstart*u.km/u.s, velend*u.km/u.s)
        # cube = fits.getdata(fitsfile+'.fits')
        cubehead = fits.getheader(fitsfile+'.fits')
        bmaj = cubehead['bmaj'] * u.deg
        bmin = cubehead['bmin'] * u.deg
        beamarea = (1.133 * bmaj * bmin).to(u.arcsec**2)
        restfreq = cubehead['restfreq'] * u.Hz
        cubewcs = WCS(cubehead)
        ra, dec = cubewcs.celestial.all_world2pix(position[0], position[1], 0)
        rah = degtohoursRA(position[0])
        dech = degtohoursDEC(position[1])
        velocities = cube.spectral_axis
        spectra = cube[:, int(dec), int(ra)]
        if kelvin:
            ax = plotSpectra(ax, velocities, spectra, freq=restfreq, beamarea=beamarea, label=label+', reg.'+regionlabel, xlims=xlims, ylims=ylims)
        else:
            ax = plotSpectra(ax, velocities, spectra, label=label+', reg.'+regionlabel, xlims=xlims, ylims=ylims)

    ax.legend(loc=0)
    # ax.set_title(r'Spectra at $\Delta$ ' +
    #              str(np.round(-deltas[0]*3600, 2))+', '+str(np.round(-deltas[1]*3600, 2))+' ('+rah+', '+dech+')')
    if saving:
        fig.savefig(fitsfiles[0]+'_delta_'+str(np.round(deltas[0]*3600, 2))+'_' + str(
            np.round(-deltas[1]*3600, 2))+'_'+specname+'.png', bbox_inches='tight')


for position, regionlabel in zip(positions, regionlabels):
    '''
    For each region in the list, plot
    '''
    fig = plt.figure(figsize=(6,4))
    plotSpectras(fig, position, fitsfilelist, labellist, regionlabel, phasecent,
                 xlims=[velstart, velend], ylims=[intmin, intmax], saving=saveaction, specname=namecomparison+'_reg'+regionlabel)
    plt.show()
    plt.clf()
