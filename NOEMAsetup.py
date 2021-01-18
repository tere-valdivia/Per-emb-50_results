import astropy.units as u
import numpy as np
from astropy.io import fits
import matplotlib as mpl
from matplotlib import rc
from astropy.constants import c, h, k_B

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

rc('font',**{'family':'serif', 'size':14})#,'sans-serif':['Helvetica']})#'family':'serif'
rc('text', usetex=True)

ra_Per50 = 15 * (3 + (29 + 7.76/60.) / 60.) * u.deg
dec_Per50 = (31 + (21 + 57.2 / 60.) / 60.) * u.deg
dist_Per50 = 293.  # pc

# Data files
H2CO_303_202 = 'H2CO/CDconfig/Per-emb-50_CD_l021l060_uvsub_H2CO_multi'
H2CO_303_202_s = 'H2CO/CDconfigsmall/Per-emb-50_CD_l021l060_uvsub_H2CO_multi_small'
SO_55_44 = 'SO_55_44/CDconfig/Per-emb-50_CD_l009l048_uvsub_SO_multi'
SO_55_44_s = 'SO_55_44/CDconfigsmall/Per-emb-50_CD_l009l048_uvsub_SO_multi_small'
SO_56_45 = 'SO_56_45/CDconfig/Per-emb-50_CD_l026l065_uvsub_SO_multi'
C18O_2_1 = 'C18O/CDconfig/Per-emb-50_CD_l025l064_uvsub_C18O_multi'

def pb_noema(freq_obs):
    """
    From jpinedaf/NOEMA_streamer_analysis
    Primary beam diameter for NOEMA at the observed frequency.
        PB = 64.1 * (72.78382*u.GHz) / freq_obs
    :param freq_obs: is the observed frequency in GHz.
    :return: The primary beam FWHM in arcsec
    """
    return (64.1 * u.arcsec * 72.78382 * u.GHz / freq_obs).decompose()

def setup_plot_noema(fig_i, label_col='black', star_col='red'):
    """
    From jpinedaf/NOEMA_streamer_analysis
    Setup of NOEMA plots, since they will show all the same format.

    """
    fig_i.set_system_latex(True)
    fig_i.ticks.set_color(label_col)
    fig_i.recenter(52.28236666667, 31.36586888889, radius=18 * (u.arcsec).to(u.deg))
    # fig_i.set_nan_color('0.9')
    fig_i.add_beam(color=label_col)
    distance = 293.
    ang_size = (2000 / distance) * u.arcsec
    fig_i.add_scalebar(ang_size, label='2000 AU', color=label_col, corner='bottom')
    fig_i.scalebar.set_label('2000 AU')
    fig_i.scalebar.set_font(family='monospace', weight=1000)
    fig_i.show_markers(ra_Per50.value, dec_Per50.value, marker='*', s=60, layer='star',
                       edgecolor=star_col, facecolor=label_col, zorder=31)
    fig_i.tick_labels.set_xformat('hh:mm:ss')
    fig_i.tick_labels.set_yformat('dd:mm:ss')
    fig_i.ticks.set_length(7)
    fig_i.axis_labels.set_xtext(r'Right Ascension (J2000)')
    fig_i.axis_labels.set_ytext(r'Declination (J2000)')
    return

def convert_into_mili(file_name):
    """
    It converts a file into one rescaled by 1e3.
    This is useful to convert between Jy -> mJy or m/s into km/s
    for plotting purposes (e.g. to use with aplpy).
    Usage:
    fig = aplpy.FITSFigure(convert_into_mili(file_in_Jy), figsize=(4,4))
    fig.show_colorscale(vmin=0, vmax=160, cmap='inferno')
    fig.add_colorbar()
    fig.colorbar.set_axis_label_text(r'Integrated Intensity (mJy beam$^{-1}$ km s$^{-1}$)')
    :param file_name: string with filename to process
    :return: hdu
    """
    data, hd = fits.getdata(file_name, header=True)
    return fits.PrimaryHDU(data=data*1e3, header=hd)
