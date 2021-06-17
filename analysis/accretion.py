import sys
sys.path.append('../')

from uncertainties import ufloat, unumpy, umath
import pickle
import velocity_tools.stream_lines as SL
from astropy.coordinates import SkyCoord, FK5
import os
from NOEMAsetup import *
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import matplotlib.pyplot as plt

'''
Important functions
'''


def distance_pix(x0, y0, x, y):
    dis = np.sqrt((x-x0)**2 + (y-y0)**2)
    return dis

def distance_pix_unumpy(x0, y0, x, y):
    square = (x-x0)**2 + (y-y0)**2
    dis = np.array([umath.sqrt(square[i]) for i in range(len(square))])
    return dis

def distance_physical(ra0, dec0, ra, dec, header):
    ra0_pix, dec0_pix = WCS(header).celestial.all_world2pix(ra0, dec0, 0)
    dist_pix = distance_pix(ra0_pix, dec0_pix, ra, dec)
    dist_deg = (np.abs(header['CDELT2']) * u.deg).to(u.arcsec) * dist_pix
    dist = dist_deg.value * dist_Per50  # pc * deg = au
    return dist


def mass_center(grid, weights):
    '''
    grid should be size (2, ny*nx)
    weights should be size (ny*nx)
    This is because they are indices obtained through np.where
    returns y,x center of mass
    '''
    totalweight = np.nansum(weights)
    valuegridy = grid[0] * weights
    valuegridx = grid[1] * weights
    centery = np.nansum(valuegridy) / totalweight
    centerx = np.nansum(valuegridx) / totalweight
    return centery, centerx


'''
Inputs
'''
Tex_u = ufloat(15, 5)
formatname = str(int(Tex_u.n)) + 'pm' + str(int(Tex_u.s))
tablefilemacc = 'M_Mdot_Tex_{0}_mom0_pbcor_binned_unc_tmodel.csv'.format(formatname)
modelname = 'H2CO_0.39Msun_env'
fileinpickle = 'streamer_model_'+modelname+'_params'
NC18Ofilename = 'N_C18O_constantTex_{0}K_mom0_pbcor.fits'
uNC18Ofilename = 'N_C18O_unc_constantTex_{0}K_mom0_pbcor.fits'
NC18Ofilenamekink = 'N_C18O_constantTex_{0}K_mom0_pbcor_kink.fits'
uNC18Ofilenamekink = 'N_C18O_unc_constantTex_{0}K_mom0_pbcor_kink.fits'
NC18Oplotname = 'N_C18O_constantTex_{0}K_mom0_pbcor.pdf'
radius3Dmapname = 'column_dens_maps/distance_3D_map.fits'
radiusmapname = 'column_dens_maps/distance_2D_map.fits'
X_C18O = 5.9e6  # Frerking et al 1982
# this is the X_C18O value used in Nishimura et al 2015 for Orion clouds
distance = (dist_Per50 * u.pc).to(u.cm)
mu_H2 = 2.7
# Texlist = np.array([10,11,12,13,14,15])* u.K
# B0 = (54891.420 * u.MHz).to(1/u.s)
binsize = 360  # au
deltar = 10*u.au  # sample size for the streamline model

# Central envelope parameters
M_s = 1.71 * u.Msun
# M_s = ufloat(1.71, 0.19)
M_env = 0.39 * u.Msun
# M_env = ufloat(0.285, 0.105)
M_disk = 0.58 * u.Msun
Mstar = (M_s + M_env + M_disk)

'''
End inputs
'''
NC18Omap = unumpy.uarray(fits.getdata('column_dens_maps/'+NC18Ofilename.format(formatname)),
                         fits.getdata('column_dens_maps/'+uNC18Ofilename.format(formatname)))
NC18Omapkink = unumpy.uarray(fits.getdata('column_dens_maps/'+NC18Ofilenamekink.format(formatname)),
                             fits.getdata('column_dens_maps/'+uNC18Ofilenamekink.format(formatname)))
NC18Oheader = fits.getheader('column_dens_maps/'+NC18Ofilename.format(formatname))
restfreq = NC18Oheader['restfreq'] * u.Hz
deltara = (NC18Oheader['CDELT1'] * u.deg).to(u.rad).value
deltadec = (NC18Oheader['CDELT2'] * u.deg).to(u.rad).value

wcsmap = WCS(NC18Oheader).celestial
leny, lenx = np.shape(NC18Omap)

# we first convert the maps to N(H2) maps
NH2map = NC18Omap * X_C18O  # * (u.cm**-2)
NH2mapkink = NC18Omapkink * X_C18O  # * (u.cm**-2)

# We create a map for physical distances from Per50 and a reference frame
leny, lenx = np.shape(NC18Omap)
yy, xx = np.mgrid[0:leny, 0:lenx]
distance_map = distance_physical(ra_Per50.value, dec_Per50.value, xx, yy, NC18Oheader)
Per50_c = SkyCoord(ra_Per50, dec_Per50, frame='fk5')
Per50_ref = Per50_c.skyoffset_frame()

# We load the model
pickle_in = open(fileinpickle+'.pickle', "rb")
streamdict = pickle.load(pickle_in)
omega0 = streamdict['omega0']
r0 = streamdict['r0']
theta0 = streamdict['theta0']
phi0 = streamdict['phi0']
v_r0 = streamdict['v_r0']
inc = streamdict['inc']
PA_ang = streamdict['PA']

# We run the streamline model
(x1, y1, z1), (vx1, vy1, vz1) = SL.xyz_stream(
    mass=Mstar, r0=r0, theta0=theta0, phi0=phi0,
    omega=omega0, v_r0=v_r0, inc=inc, pa=PA_ang, rmin=10*u.au, deltar=deltar)
rc = SL.r_cent(mass=Mstar, omega=omega0, r0=r0)

# we need the pixel location of the streamer for the future
dra_stream = -x1.value / dist_Per50
ddec_stream = z1.value / dist_Per50
fil = SkyCoord(dra_stream*u.arcsec, ddec_stream*u.arcsec,
               frame=Per50_ref).transform_to(FK5)
xstream, zstream = wcsmap.all_world2pix(fil.ra.value, fil.dec.value, 0)

# (x1, y1, z1) is in au
mass_streamer_table = pd.DataFrame()

dist_streamer = np.sqrt(x1**2+y1**2+z1**2)  # distance from protostar to point
dist_projected = np.sqrt(x1**2+z1**2)  # distance from protostar to point in the sky
vel_streamer = np.sqrt(vx1**2+vy1**2+vz1**2)  # velocity at point x,y,z

# we calculate the size of each segment and the time to travel the streamer
dx1 = np.roll(x1, 1) - x1
dy1 = np.roll(y1, 1) - y1
dz1 = np.roll(z1, 1) - z1

# now, we take the mean velocity between segments and calculate deltat
deltas = np.sqrt(dx1**2 + dy1**2 + dz1**2)[1:]
vel_mean = 0.5 * (np.roll(vel_streamer, 1) + vel_streamer)[1:]
dist_mean = 0.5 * (np.roll(dist_streamer, 1) + dist_streamer)[1:]
dist_projected_mean = 0.5 * (np.roll(dist_projected, 1) + dist_projected)[1:]
deltat = (deltas / vel_mean).to(u.yr)

# usually we need to sacrifice the first two points: the first because of the
# roll and the second because of the initial v=0. At this point is important
# to check how many points we need to leave out, specially with omega0 close
# to 1e-14 / u.s, and modify the range above

# we reverse two times to sum from protostar to start point and return the
# array to its original order
time_integral_path = np.flip(np.cumsum(np.flip(deltat))).to(u.yr)
time_theory = t_freefall_acc(0 * u.au, dist_mean, r0, mass=Mstar)

# Now, we separate the streamer in bins
# for the streamer-calculated distances
# TODO: Implement the rolling average to sample the full streamer
radiuses = np.arange(0., 3200., binsize)  # list of projected lengths we want to sample
binradii = np.zeros(len(radiuses)-1) * u.AU  # length of the streamer in bin
binradiierr = np.zeros(len(radiuses)-1) * u.AU
binradiikink = np.zeros(len(radiuses)-1) * u.AU
binradiikinkerr = np.zeros(len(radiuses)-1) * u.AU
masses = unumpy.uarray(np.zeros(len(binradii)), np.zeros(len(binradii)))
masseskink = unumpy.uarray(np.zeros(len(binradii)), np.zeros(len(binradii)))
# times = np.zeros(len(binradii)) * u.yr
times = unumpy.uarray(np.zeros(len(binradii)), np.zeros(len(binradii)))
timeskink = unumpy.uarray(np.zeros(len(binradii)), np.zeros(len(binradii)))
times_theory = np.zeros(len(binradii)) * u.yr
m_acclist = unumpy.uarray(np.zeros(len(binradii)), np.zeros(len(binradii)))
m_acclistkink = unumpy.uarray(np.zeros(len(binradii)), np.zeros(len(binradii)))

mass_streamer_table['2D bin minimum (au)'] = radiuses[:len(radiuses)-1]
mass_streamer_table['2D bin maximum (au)'] = radiuses[1:]
mask = np.where((distance_map > radiuses[0]) & (distance_map < radiuses[1]) & ~np.isnan(unumpy.nominal_values(NH2map)))
for i in range(len(radiuses)-1):
    # we want to make a map of the bins, so we need to save the indexes
    # of the map positions
    mask = np.where((distance_map > radiuses[i]) & (distance_map < radiuses[i+1]) & ~np.isnan(unumpy.nominal_values(NH2map)))
    maskkink = np.where((distance_map > radiuses[i]) & (distance_map < radiuses[i+1]) & ~np.isnan(unumpy.nominal_values(NH2mapkink)))
    weight_map = NH2map[mask]  # recall these have uncertainties
    weight_mapkink = NH2mapkink[maskkink]
    centerz, centerx = mass_center(mask, weight_map)
    centerzkink, centerxkink = mass_center(maskkink, weight_mapkink)
    # streamer distances within these bin of projected distances
    # indexbin is for the array of the streamer
    indexbin = np.where((dist_projected_mean.value > radiuses[i]) & (
        dist_projected_mean.value < radiuses[i+1]))
    streamer_distances = dist_mean[indexbin]
    streamer_times = time_integral_path[indexbin]
    streamer_times_theory = time_theory[indexbin]
    xs = xstream[indexbin]
    zs = zstream[indexbin]

    # indexsol is the index we need to fill the table
    distancepix = distance_pix_unumpy(centerx, centerz, xs, zs)
    distancepixkink = distance_pix_unumpy(centerxkink, centerzkink, xs, zs)
    indexsol = np.argmin(distancepix)
    indexsolerr = np.argmin(unumpy.nominal_values(distancepix) + unumpy.std_devs(distancepix))
    indexsolkink = np.argmin(distancepixkink)
    indexsolkinkerr = np.argmin(unumpy.nominal_values(distancepixkink) + unumpy.std_devs(distancepixkink))

    binradii[i] = streamer_distances[indexsol]
    binradiierr[i] = np.abs(streamer_distances[indexsol] - streamer_distances[indexsolerr]) if indexsol != indexsolerr else deltar
    binradiikink[i] = streamer_distances[indexsolkink]
    binradiikinkerr[i] = np.abs(streamer_distances[indexsolkink] - streamer_distances[indexsolkinkerr]) if indexsolkink != indexsolkinkerr else deltar
    # times[i] = streamer_times[indexsol]
    timeserr = np.amax([np.abs(streamer_times[indexsol] - streamer_times[indexsolerr]).value, np.abs(streamer_times[indexsol] - streamer_times[indexsol+1]).value])
    times[i] = ufloat(streamer_times[indexsol].value, timeserr)
    timeskinkerr = np.amax([np.abs(streamer_times[indexsolkink] - streamer_times[indexsolkinkerr]).value, np.abs(streamer_times[indexsolkink] - streamer_times[indexsolkink+1]).value])
    timeskink[i] = ufloat(streamer_times[indexsolkink].value, timeskinkerr)

    times_theory[i] = streamer_times_theory[indexsol]

    NH2tot = np.sum(weight_map)
    NH2totkink = np.sum(weight_mapkink)

    masses[i] = M_hydrogen2(NH2tot, mu_H2, distance, deltara, deltadec)
    masseskink[i] = M_hydrogen2(NH2totkink, mu_H2, distance, deltara, deltadec)

    m_acclist[i] = masses[i] / times[i]
    m_acclistkink[i] = masseskink[i] / timeskink[i]

mass_streamer_table['3D distance (au)'] = binradii
mass_streamer_table['u 3D distance (au)'] = binradiierr
mass_streamer_table['3D distance kink (au)'] = binradiikink
mass_streamer_table['u 3D distance kink (au)'] = binradiikinkerr
mass_streamer_table['Mass wo kink (Msun)'] = unumpy.nominal_values(masses)
mass_streamer_table['u Mass wo kink (Msun)'] = unumpy.std_devs(masses)
mass_streamer_table['Mass w kink (Msun)'] = unumpy.nominal_values(masseskink)
mass_streamer_table['u Mass w kink (Msun)'] = unumpy.std_devs(masseskink)
mass_streamer_table['t_ff streamline (yr, M_env = 0.39 Msun)'] = unumpy.nominal_values(times)
mass_streamer_table['u t_ff streamline (yr, M_env = 0.39 Msun)'] = unumpy.std_devs(times)
mass_streamer_table['t_ff streamline kink (yr, M_env = 0.39 Msun)'] = unumpy.nominal_values(timeskink)
mass_streamer_table['u t_ff streamline kink (yr, M_env = 0.39 Msun)'] = unumpy.std_devs(timeskink)
mass_streamer_table['t_ff theory (yr, M_env = 0.39 Msun)'] = times_theory
mass_streamer_table['Mdot wo kink (Msun yr-1)'] = unumpy.nominal_values(m_acclist)
mass_streamer_table['u Mdot wo kink (Msun yr-1)'] = unumpy.std_devs(m_acclist)
mass_streamer_table['Mdot w kink (Msun yr-1)'] = unumpy.nominal_values(m_acclistkink)
mass_streamer_table['u Mdot w kink (Msun yr-1)'] = unumpy.std_devs(m_acclistkink)
if not os.path.exists(tablefilemacc):
    mass_streamer_table.to_csv(tablefilemacc)