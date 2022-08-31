'''
Author: Teresa Valdivia-Mena
Last revised August 31, 2022

This code is set to estimate the mass and infall rate along the streamer,
 given than the column density maps from mass_estimation_C18O.py already exist

Current state: 0.18 Msun
'''
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

def M_hydrogen2_unumpy(N, mu, D, deltara, deltadec):
    '''
    D must be in cm
    '''
    preamble = (mu * m_p) * np.abs(deltara * deltadec)
    Mass = N * (D**2) * preamble.value
    Mass_unit = ((1. * preamble.unit).to(u.Msun)).value
    # return Mass.to(u.Msun)
    return Mass * Mass_unit
'''
Inputs
'''
Tex_u = ufloat(15, 5)
Menv = 0.18 #0.18 M_sun
# Tex_u = 15
formatname = str(int(Tex_u.n)) + 'pm' + str(int(Tex_u.s))

tablefilemacc = 'M_Mdot_Tex_{0}_mom0_pbcor_rolledbin_unc_tmodel_Menv_{1}Msun.csv'.format(formatname, Menv)
modelname = 'H2CO_{0}Msun_env'.format(Menv)
fileinpickle = 'streamer_model_'+modelname+'_params'
NC18Ofilenamekink = 'N_C18O_constantTex_{0}K_mom0_pbcor_kink.fits'
uNC18Ofilenamekink = 'N_C18O_unc_constantTex_{0}K_mom0_pbcor_kink.fits'
NC18Oplotname = 'N_C18O_constantTex_{0}K_mom0_pbcor.pdf'
X_C18O = 5.9e6  # Frerking et al 1982
# this is the X_C18O value used in Nishimura et al 2015 for Orion clouds
distance = (dist_Per50 * u.pc).to(u.cm).value
# distance = ufloat((dist_Per50 * u.pc).to(u.cm).value, (22*u.pc).to(u.cm).value)
mu_H2 = 2.7
binsize = 360  # au
deltar = 10 * u.au  # sample size for the streamline model

# Central envelope parameters
M_s = 1.71 * u.Msun
# M_s = ufloat(1.71, 0.19)
M_env = Menv * u.Msun
# M_env = ufloat(0.285, 0.105)
M_disk = 0.58 * u.Msun
Mstar = (M_s + M_env + M_disk)

'''
End inputs
'''
NC18Omapkink = unumpy.uarray(fits.getdata('column_dens_maps/'+NC18Ofilenamekink.format(formatname)),
                             fits.getdata('column_dens_maps/'+uNC18Ofilenamekink.format(formatname)))
NC18Oheader = fits.getheader('column_dens_maps/'+NC18Ofilenamekink.format(formatname))
restfreq = NC18Oheader['restfreq'] * u.Hz
deltara = (NC18Oheader['CDELT1'] * u.deg).to(u.rad).value
deltadec = (NC18Oheader['CDELT2'] * u.deg).to(u.rad).value

wcsmap = WCS(NC18Oheader).celestial
leny, lenx = np.shape(NC18Omapkink)

# we first convert the maps to N(H2) maps
NH2mapkink = NC18Omapkink * X_C18O  # * (u.cm**-2)

# We create a map for physical distances from Per50 and a reference frame
leny, lenx = np.shape(NC18Omapkink)
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
rc = SL.r_cent(mass=Mstar, omega=omega0, r0=r0)
(x1, y1, z1), (vx1, vy1, vz1) = SL.xyz_stream(
    mass=Mstar, r0=r0, theta0=theta0, phi0=phi0,
    omega=omega0, v_r0=v_r0, inc=inc, pa=PA_ang, rmin=rc) #, deltar=deltar)


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
# usually we need to sacrifice the first two points: the first because of the
# roll and the second because of the initial v=0. At this point is important
# to check how many points we need to leave out, specially with omega0 close
# to 1e-14 / u.s, and modify the range below
deltas = np.sqrt(dx1**2 + dy1**2 + dz1**2)[1:] # this is delta S, the path
vel_mean = 0.5 * (np.roll(vel_streamer, 1) + vel_streamer)[1:]
dist_mean = 0.5 * (np.roll(dist_streamer, 1) + dist_streamer)[1:]
dist_projected_mean = 0.5 * (np.roll(dist_projected, 1) + dist_projected)[1:]
deltat = (deltas / vel_mean).to(u.yr)
indexsum = np.where(dist_mean>rc)

# we reverse two times to sum from protostar to start point and return the
# array to its original order
time_integral_path = np.flip(np.cumsum(np.flip(deltat))).to(u.yr)
print('The total infall time is '+str(time_integral_path[0]))

# Now, we separate the streamer in bins
# for the streamer-calculated distances
radiuses = np.arange(200, 3200.-binsize, binsize/2)  # list of projected lengths we want to sample
radiuses2 = np.arange(200+binsize, 3200., binsize/2)

binradiikink = np.zeros(len(radiuses)) * u.AU # length of the streamer in bin
binradiikinkerr = np.zeros(len(radiuses)) * u.AU
masseskink = unumpy.uarray(np.zeros(len(binradiikink)), np.zeros(len(binradiikink)))
timeskink = unumpy.uarray(np.zeros(len(binradiikink)), np.zeros(len(binradiikink)))
m_inlistkink = unumpy.uarray(np.zeros(len(binradiikink)), np.zeros(len(binradiikink)))
deltatlist = np.zeros(len(radiuses)) * u.yr
mass_streamer_table['2D bin minimum (au)'] = radiuses
mass_streamer_table['2D bin maximum (au)'] = radiuses2

for i in range(len(radiuses)):
    # we want to make a map of the bins, so we need to save the indexes
    # of the map positions
    maskkink = np.where((distance_map > radiuses[i]) & (distance_map < radiuses2[i]) & ~np.isnan(unumpy.nominal_values(NH2mapkink)))
    weight_mapkink = NH2mapkink[maskkink]
    centerzkink, centerxkink = mass_center(maskkink, weight_mapkink)
    # streamer distances within these bin of projected distances
    # indexbin is for the array of the streamer
    indexbin = np.where((dist_projected_mean.value > radiuses[i]) & (
        dist_projected_mean.value < radiuses2[i]))
    streamer_distances = dist_mean[indexbin]
    streamer_times = time_integral_path[indexbin]
    deltat_bin = deltat[indexbin]
    xs = xstream[indexbin]
    zs = zstream[indexbin]

    # indexsol is the index we need to fill the table
    distancepixkink = distance_pix_unumpy(centerxkink, centerzkink, xs, zs)
    indexsolkink = np.argmin(distancepixkink)
    indexsolkinkerr = np.argmin(unumpy.nominal_values(distancepixkink) + unumpy.std_devs(distancepixkink))

    binradiikink[i] = streamer_distances[indexsolkink]
    binradiikinkerr[i] = np.abs(streamer_distances[indexsolkink] - streamer_distances[indexsolkinkerr]) if indexsolkink != indexsolkinkerr else deltar

    timeskinkerr = np.amax([np.abs(streamer_times[indexsolkink] - streamer_times[indexsolkinkerr]).value, np.abs(streamer_times[indexsolkink] - streamer_times[indexsolkink+1]).value])
    timeskink[i] = ufloat(streamer_times[indexsolkink].value, timeskinkerr)

    NH2totkink = np.sum(weight_mapkink)
    masseskink[i] = M_hydrogen2_unumpy(NH2totkink, mu_H2, distance, deltara, deltadec)
    deltatlist[i] = np.sum(deltat_bin)
    m_inlistkink[i] = masseskink[i] / deltatlist[i].value

mass_streamer_table['3D distance kink (au)'] = binradiikink
mass_streamer_table['u 3D distance kink (au)'] = binradiikinkerr
mass_streamer_table['Mass w kink (Msun)'] = unumpy.nominal_values(masseskink)
mass_streamer_table['u Mass w kink (Msun)'] = unumpy.std_devs(masseskink)
mass_streamer_table['t_ff streamline kink (yr, M_env = {0} Msun)'.format(Menv)] = unumpy.nominal_values(timeskink)
mass_streamer_table['u t_ff streamline kink (yr, M_env = {0} Msun)'.format(Menv)] = unumpy.std_devs(timeskink)
mass_streamer_table['deltat_ff streamline (yr, M_env = {0} Msun)'.format(Menv)] = deltatlist.value
mass_streamer_table['Mdot_in w kink (Msun yr-1)'] = unumpy.nominal_values(m_inlistkink)
mass_streamer_table['u Mdot_in w kink (Msun yr-1)'] = unumpy.std_devs(m_inlistkink)
if not os.path.exists(tablefilemacc):
    mass_streamer_table.to_csv(tablefilemacc)

NH2tot = np.nansum(NH2mapkink)
MH2tot = M_hydrogen2_unumpy(NH2tot, mu_H2, distance, deltara, deltadec)
timetot = time_integral_path[0].value
print(MH2tot/timetot)
