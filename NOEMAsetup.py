import astropy.units as u
import numpy as np
from astropy.io import fits
import matplotlib as mpl
from matplotlib import rc
from astropy.constants import c, h, k_B, m_p, G
from uncertainties import ufloat, umath

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

rc('font', **{'family': 'serif', 'size': 14})  # ,'sans-serif':['Helvetica']})#'family':'serif'
rc('text', usetex=True)

ra_Per50 = 15 * (3 + (29 + 7.76/60.) / 60.) * u.deg
dec_Per50 = (31 + (21 + 57.2 / 60.) / 60.) * u.deg
dist_Per50 = 293.  # pc

# Data files for imaging
H2CO_303_202 = 'H2CO/CDconfig/Per-emb-50_CD_l021l060_uvsub_H2CO_multi'
H2CO_303_202_s = 'H2CO/CDconfigsmall/Per-emb-50_CD_l021l060_uvsub_H2CO_multi_small'
H2CO_303_202_s_pb = 'H2CO/CDconfigsmall/Per-emb-50_CD_l021l060_uvsub_H2CO_multi_pbcor_small'
SO_55_44 = 'SO_55_44/CDconfig/Per-emb-50_CD_l009l048_uvsub_SO_multi'
SO_55_44_s = 'SO_55_44/CDconfigsmall/Per-emb-50_CD_l009l048_uvsub_SO_multi_small'
SO_56_45 = 'SO_56_45/CDconfig/Per-emb-50_CD_l026l065_uvsub_SO_multi'
C18O_2_1 = 'C18O/CDconfig/JEP/JEP_mask_multi_Per-emb-50_CD_l025l064_uvsub_C18O'
C18O_2_1_s = 'C18O/CDconfig/JEP/JEP_mask_multi_Per-emb-50_CD_l025l064_uvsub_C18O'


# Data files for analysis
H2CO_303_202_TdV = 'data/Per-emb-50_CD_l021l060_uvsub_H2CO_multi_total_mom0'
H2CO_303_202_TdV_s = 'data/Per-emb-50_CD_l021l060_uvsub_H2CO_multi_small_fitcube_stream_mom0'
C18O_2_1_TdV = 'data/JEP_mask_multi_Per-emb-50_CD_l025l064_uvsub_C18O_fitcube_total_mom0'
C18O_2_1_fit_Vc = 'data/JEP_mask_multi_Per-emb-50_CD_l025l064_uvsub_C18O_1G_Vc'
C18O_2_1_fit_Vc_pb = 'data/JEP_mask_multi_Per-emb-50_CD_l025l064_uvsub_C18O_pbcor_1G_Vc'
H2CO_303_202_fit_Vc = 'data/Per-emb-50_CD_l021l060_uvsub_H2CO_multi_small_1G_Vc'
H2CO_303_202_pb_fit_Vc = 'data/Per-emb-50_CD_l021l060_uvsub_H2CO_multi_pbcor_small_1G_Vc'
region_streamer = 'data/region_streamer.reg'
region_streamer_s = 'data/region_streamer_s.reg'

CO21blue = 'data/Per50.12CO21.robust-1_1.0to4.5_blue'
CO21red = 'data/Per50.12CO21.robust-1_10.5to14.0_red'


def pb_noema(freq_obs):
    """
    From jpinedaf/NOEMA_streamer_analysis
    Primary beam diameter for NOEMA at the observed frequency.
        PB = 64.1 * (72.78382*u.GHz) / freq_obs
    :param freq_obs: is the observed frequency in GHz.
    :return: The primary beam FWHM in arcsec
    """
    return (64.1 * u.arcsec * 72.78382 * u.GHz / freq_obs).decompose()


def pb_sma(freq_obs):
    """
    Primary beam diameter for SMA at the observed frequency.
        PB = 48.0 * (231.0*u.GHz) / freq_obs
    :param freq_obs: is the observed frequency in GHz.
    :return: The primary beam FWHM in arcsec
    """
    return (48.0 * u.arcsec * 231 * u.GHz / freq_obs).decompose()


def setup_plot_noema(fig_i, label_col='black', star_col='red'):
    """
    Setup of NOEMA plots, since they will show all the same format.

    Taken from jpinedaf/NOEMA_streamer_analysis, this function takes an
    aplpy FITSFigure and adds a beam, a star where Per-emb-50 is, a
    scalebar of 2000 AU and sets the format of the axes.

    Args:
        fig_i (aplpy.FITSFigure): figure to setup

        label_col (string or RGB color): color for the labels (star filling and
                                        scalebar colors)
        star_col (string or RGB color): color for the star edge

    """
    fig_i.set_system_latex(True)
    fig_i.ticks.set_color(label_col)
    # fig_i.recenter(52.28236666667, 31.36586888889, radius=18 * (u.arcsec).to(u.deg))
    # fig_i.set_nan_color('0.9')
    fig_i.add_beam(color=label_col)
    distance = 293.
    ang_size = (2000 / distance) * u.arcsec
    fig_i.add_scalebar(ang_size, label='2000 au', color=label_col, corner='bottom')
    fig_i.scalebar.set_label('2000 au')
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


def per_emb_50_get_vc_r(velfield_file, region_file):
    """
    Returns the centroid velocity and projected separation in the sky for the
    centroid velocity from Per-emb-50

    Given a region and a velocity field for the vicinity of Per-emb-50,
    obtains the projected radius and the central velocity of each pixel in the
    region. The velocity field must be masked to contain only the relevant
    pixels.

    Args:
        velfield_file (string): path to the .fits file containing the velocity
        field
        region_file (string): path to the .reg (ds9) region file where the
        streamer is contained

    Returns:
        type: description

    """
    from regions import read_ds9
    from astropy.wcs import WCS
    from astropy.io import fits
    import velocity_tools.coordinate_offsets as c_offset
    # load region file and WCS structures
    regions = read_ds9(region_file)
    wcs_Vc = WCS(velfield_file)
    #
    hd_Vc = fits.getheader(velfield_file)
    results = c_offset.generate_offsets(
        hd_Vc, ra_Per50, dec_Per50, pa_angle=0*u.deg, inclination=0*u.deg)
    rad_au = (results.r * dist_Per50*u.pc).to(u.au, equivalencies=u.dimensionless_angles())
    # Vc_all =
    #
    mask_Vc = (regions[0].to_pixel(wcs_Vc)).to_mask()
    Vc_cutout = mask_Vc.cutout(fits.getdata(velfield_file))
    rad_cutout = mask_Vc.cutout(rad_au)
    #
    gd = (mask_Vc.data == 1)
    v_los = Vc_cutout[gd]*u.km/u.s
    r_proj = rad_cutout[gd]
    return r_proj, v_los

def t_freefall(r, M):
    '''
    r must be a AU Quantity
    M must be a M_sun Quantity
    Returns free-fall time in yr Quantity
    '''
    t= np.sqrt((r)**3/(G * M)).decompose().to(u.yr) * np.pi/(np.sqrt(2)*2)
    return t

def t_freefall_unumpy(r, M):
    '''
    r must be a AU Quantity
    M must be a M_sun ufloat (no quantity)
    Returns free-fall time in yr
    '''
    preamble = np.sqrt((r)**3/(G)).decompose().to(u.yr*u.Msun**(1./2))
    t = preamble.value / umath.sqrt(M) * np.pi/(np.sqrt(2)*2)
    return t

def t_freefall_acc(r_fin, r_init, r0, mass=1*u.Msun):
  """
  Returns the freefall timescale along a path in yr

  Based on t_freefall, we calculate the integral between r0 and r
  This equation considers that v_r in r_init is 0, and the velocity is
  non-zero in the inner points
  r_fin is the array between 0 and r' \leq r0
  """
  eps_fin = (r_fin/r0).value
  eps_init = (r_init/r0).value
  theta_fin = np.arcsin(np.sqrt(eps_fin))
  theta_init = np.arcsin(np.sqrt(eps_init)) #np.pi/2 #  arcsin(1) = np.pi/2
  integral = np.sqrt(r0**3/(2 * G * mass)) * ((theta_init - np.sin(theta_init) * np.cos(theta_init)) - (theta_fin - np.sin(theta_fin) * np.cos(theta_fin)))
  return integral.to(u.yr)


def t_freefall_acc_unumpy(r_fin, r_init, r0, mass):
  """
  Returns the freefall timescale along a path in yr

  Based on t_freefall, we calculate the integral between r0 and r
  This equation considers that v_r in r_init is 0, and the velocity is
  non-zero in the inner points
  r_fin is the array between 0 and r' \leq r0

  Optimized for unumpy
  """
  eps_fin = (r_fin/r0).value
  eps_init = (r_init/r0).value
  theta_fin = np.arcsin(np.sqrt(eps_fin))
  theta_init = np.arcsin(np.sqrt(eps_init)) #np.pi/2 #  arcsin(1) = np.pi/2
  constant = np.sqrt(r0**3/(2 * G)).to(u.yr * (u.Msun)**(1/2))
  integral = constant.value / umath.sqrt(mass) * ((theta_init - np.sin(theta_init) * np.cos(theta_init)) - (theta_fin - np.sin(theta_fin) * np.cos(theta_fin)))
  return integral


def M_hydrogen2(N, mu, D, deltara, deltadec):
    """
    Returns the gas mass of molecular hydrogen, given the sum of column density
    N in cm-2 Npix.

    Requires N to be in cm-2 but without unit. Returns in solar masses

    Args:
        N (ndarray): sum of the column densities in cm-2 Npix
        mu (float): molecular weight of the hydrogen atom
        D (Quantity (pc)): distance to the source
        deltara, deltadec (float): size of the pixel in radians
    Returns:
        mass (float): total mass in solar masses

    """
    preamble = (mu * m_p) * (D**2) * np.abs(deltara * deltadec)
    Mass = N * preamble.value
    Mass_unit = ((1. * u.cm**(-2) * preamble.unit).to(u.Msun)).value
    # return Mass.to(u.Msun)
    return Mass * Mass_unit
