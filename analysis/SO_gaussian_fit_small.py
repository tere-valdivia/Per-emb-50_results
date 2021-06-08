import numpy as np
from scipy import stats
from spectral_cube import SpectralCube
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.modeling.functional_models import Gaussian1D
import os
import sys
sys.path.append('../')
from NOEMAsetup import *
import pyspeckit
import regions
from regions import PixCoord
import matplotlib.pyplot as plt

'''
Note that we are NOT USING A PB CORRECTED CUBE
'''

def chi_square(yPred, yData, err):
    chi2 = np.sum((yPred-yData)**2/(err**2))
    return chi2

# We use the AIC definition used in Choudhury et al 2020 (with least square
# instead of the maximum likelihood)
def AIC(yPred, data, err, k):
    """
    Returns the Akaike information criterion (AIC) for a given function with a
    number of parameters k and a negative log-likelihood value given
    by func(data, params)
    """
    # ll = log_likelihood(yPred, data, err)
    # aic = 2 * k + 2 * ll
    chi2 = chi_square(yPred, data, err)
    aic = 2 * k + chi2 # we leave out the constant because it is the same for
    # both models
    return aic

def probaic(aicmin, aiclist):
    return np.exp((aicmin-aiclist)/2)


def filter1G(spc, rms, rmslevel, errorfrac=0.5, epsilon=1.e-5,negative=True):
    """
    Replace the pixels in the fitted cube with np.nan where the fit is not
    good enough according to our criteria.

    The criteria that a pixel must have are:
    - The error is not zero
    - The error fraction is lower than errorfrac
    - If one pixel in a spectra is np.nan, all the spectra must be nan (sanity
    check)
    - All peaks must have S/N>rmslevel

    Args:
        variable (type): description

    Returns:
        type:

    """
    zeromask = np.where(np.abs(spc.errcube[0]) < epsilon, 1, 0) + \
        np.where(np.abs(spc.errcube[1]) < epsilon, 1, 0) + \
        np.where(np.abs(spc.errcube[2]) < epsilon, 1, 0)
    spc.parcube[np.where(np.repeat([zeromask], 3, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([zeromask], 3, axis=0))] = np.nan
    if negative:
        negativemask = np.where(spc.parcube[0] < 0, 1, 0) + \
            np.where(spc.parcube[1] < 0, 1, 0) + \
            np.where(spc.parcube[2] < 0, 1, 0)
        spc.parcube[np.where(np.repeat([negativemask], 3, axis=0))] = np.nan
        spc.errcube[np.where(np.repeat([negativemask], 3, axis=0))] = np.nan
    errormask = np.where(np.abs(spc.errcube[0]/spc.parcube[0]) > errorfrac, 1, 0)\
        + np.where(np.abs(spc.errcube[1]/spc.parcube[1]) > errorfrac, 1, 0)\
        + np.where(np.abs(spc.errcube[1]/spc.parcube[1]) > errorfrac, 1, 0)
    spc.parcube[np.where(np.repeat([errormask], 3, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([errormask], 3, axis=0))] = np.nan
    # Force if one pixel in a channel is nan, all the same pixels
    # in all  channels must be nan
    nanmask = np.sum(np.where(np.isnan(np.concatenate([spc.parcube, spc.errcube])), 1, 0), axis=0)
    spc.parcube[np.where(np.repeat([nanmask], 3, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([nanmask], 3, axis=0))] = np.nan
    peakmask = np.where(spc.parcube[0] < rmslevel*rms, 1, 0)
    spc.parcube[np.where(np.repeat([peakmask], 3, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([peakmask], 3, axis=0))] = np.nan
    return spc


'''
We use the following assumption:
 "If the model residuals are expected to be normally distributed then a
 log-likelihood function based on the one above can be used." In R notation
 this is for a linear regression:
 LL <- function(beta0, beta1, mu, sigma) {
    R = y - x * beta1 - beta0
    #
    R = suppressWarnings(dnorm(R, mu, sigma, log = TRUE))
    #
    -sum(R)
}
'''
########
# Inputs
########
cubefile = '../SO_55_44/CDconfigsmall/Per-emb-50_CD_l009l048_uvsub_SO_multi_small'
velinit = -1.0 * u.km/u.s
velend = 13.0 * u.km/u.s
starting_point = (137,115)
prob_tolerance = 0.05
snratio = 4 # minimal signal to noise ratio for the fitted amplitude
signal_cut = 4 # minimal signal to noise the data must have to fit
min_sigma = 0.08 # 1 channel
########
# End Inputs
########


cube = SpectralCube.read(cubefile+'.fits')
cube = cube.to(u.K)
if not os.path.exists(cubefile+'_K.fits'):
    cube.hdu.writeto(cubefile+'_K.fits')
cube = cube.with_spectral_unit(u.km/u.s)
velaxis = cube.spectral_axis
n_y, n_x = np.shape(cube.unmasked_data[0,:,:])
spc = pyspeckit.Cube(cube=cube)
header = cube.header
freq = (header['RESTFREQ']/1e9) * u.GHz
wcsspec = WCS(header).spectral
wcscel = WCS(header).celestial

# measure the noise only where there is no line emission
chanlims = [wcsspec.world_to_pixel(velinit).tolist(), wcsspec.world_to_pixel(velend).tolist()]
rms = np.nanstd(np.vstack([cube[:int(np.min(chanlims))], cube[int(np.max(chanlims)):]]))
rmsmap = np.ones(np.shape(spc.cube)) * rms
spc.errorcube = rmsmap

momentsfile = cubefile+'_moments.fits'
cube_mom = cube.spectral_slab(velinit,velend)
spc_mom = pyspeckit.Cube(cube=cube_mom)

if os.path.exists(momentsfile):
    spc.momentcube = fits.getdata(momentsfile)
else:
    spc_mom.momenteach(negamp=False, vheight=False)
    spc.momentcube = spc_mom.momentcube
    moments = fits.PrimaryHDU(data=spc.momentcube, header=wcscel.to_header())
    moments.writeto(momentsfile)
spc2 = spc.copy()
spc3 = spc.copy()
# We use the moments to do initial guesses
# Use the skew to determine the direction of the second gaussian


"""
1 gaussian
"""
# mom01 = np.where(spc.momentcube[0]>rms*snratio, spc.momentcube[0],rms*snratio)
# initguesses = np.array([mom01,spc.momentcube[1],spc.momentcube[2]])

mom01 = np.where(spc.momentcube[0]>rms*snratio, spc.momentcube[0],rms*snratio)
leny, lenx = np.shape(mom01)
initguesses = np.array([mom01,spc.momentcube[1],spc.momentcube[2]])
fitfile1 = cubefile + '_1G_fitparams.fits'
if not os.path.exists(fitfile1):
    try:
        spc.fiteach(fittype='gaussian',
                    guesses=initguesses,
                    # negamp=False,
                    verbose=1,
                    signal_cut=signal_cut,
                    blank_value=np.nan,
                    start_from_point=(starting_point))
    # except ValueError:
    #     print('There are non-finite parameters in the fit')
    except AssertionError:
        print('There are non-finite parameters in the fit')
    spc.write_fit(fitfile1)
else:
    spc.load_model_fit(fitfile1, 3, fittype='gaussian')

'''
The pixels we sample are:

We need to choose pixels to sample
'''
# plot_x = [25,23,29,37,41,30,33,40,34]
# plot_y = [34,36,25,23,13,12,41,18,24]
# for x, y in zip(plot_x,plot_y):
#     sp = spc.get_spectrum(x,y)
#     sp.plotter()
#     try:
#         sp.specfit.plot_fit()
#     except ValueError:
#         print("{0}, {1} has no fit for 2G".format(x, y))
#     sp.plotter.savefig(cubefile+ '_1G_'+str(x)+'_'+str(y)+'.png')
#     plt.close('all')

spc = filter1G(spc,rms,snratio)
# save filtered map
newheadaic = wcscel.to_header()
fitfile1filtered = cubefile + '_1G_fitparams_filtered.fits'
if not os.path.exists(fitfile1filtered):
    spc.write_fit(fitfile1filtered)

fittedmodel1 = spc.get_modelcube()

# we need to put the small fit with the streamer in the corresponding pixels
fitcubefile = '../SO_55_44/CDconfigsmall/gaussian_fit_123G_fitcube2g/Per-emb-50_CD_l009l048_uvsub_SO_multi_small_gaussian_streamer_model.fits'
fitregionfile = 'SO_2G_fitregion.reg'

regionlist = regions.read_ds9(fitregionfile)
region_sample = regionlist[0].to_pixel(wcscel)
region_mask = region_sample.to_mask().to_image((leny, lenx))


# we load the fitcube
fitcube = fits.getdata(fitcubefile)[0:3]
fitcubeshape = np.shape(fitcube)
initx = int(region_sample.center.x - region_sample.width/2)
inity = int(region_sample.center.y - region_sample.height/2)
for k in range(3):
    for j in range(int(region_sample.height)):
        for i in range(int(region_sample.width)):
            spc.parcube[k,j+inity,i+initx] = fitcube[k, j, i]


# for y,x,j,i in zip(wheremask[1],wheremask[2], range(fitcubeshape[1]),range(fitcubeshape[2])):
#     amplitude[y,x] = fitcube[0,j,i]
fitfile1streamer = cubefile + '_1G_streamer.fits'
if not os.path.exists(fitfile1streamer):
    spc.write_fit(fitfile1streamer)
