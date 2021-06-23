import sys
sys.path.append('../')

import numpy as np
import astropy.units as u
import velocity_tools.stream_lines as SL
import matplotlib.pyplot as plt
from matplotlib import animation
from astropy.wcs import WCS
from astropy.io import fits
from NOEMAsetup import *
from astropy.coordinates import SkyCoord, FK5
import pyregion
import pickle


modelname = 'H2CO_0.39Msun_env'
filenamepickle = 'analysis/streamer_model_'+modelname+'_params.pickle'

M_s = 1.71*u.Msun
M_env = 0.39*u.Msun
M_disk = 0.58*u.Msun
Mstar = (M_s+M_env+M_disk)

pickle_in = open(filenamepickle, "rb")
streamdict = pickle.load(pickle_in)
omega0 = streamdict['omega0']
r0 = streamdict['r0']
theta0 = streamdict['theta0']
phi0 = streamdict['phi0']
v_r0 = streamdict['v_r0']
inc = streamdict['inc']
PA_ang = streamdict['PA']
(x1, y1, z1), (vx1, vy1, vz1) = SL.xyz_stream(
    mass=Mstar, r0=r0, theta0=theta0, phi0=phi0,
    omega=omega0, v_r0=v_r0, inc=inc, pa=PA_ang, rmin=10*u.au, deltar=10*u.au)



fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# ax.plot(x1, y1, z1, marker='o', markersize=1)
# ax.plot(x1[0], y1[0], z1[0], marker='o', color='k',label='Start')
# ax.plot(0, 0, 0, marker='o', color='r', label='Protostar')
# ax.set_xlim([0,3000])
# ax.set_ylim([0,3000])
# ax.set_zlim([-3000,0])
# ax.legend()
# plt.show()

# we can also add a movie

def init():
    ax.set_xlabel('Offset RA (pc)')
    ax.set_ylabel('Line of sight distance (pc)')
    ax.set_zlabel('Offset DEC (pc)')
    ax.plot(x1, y1, z1, marker='o', markersize=1)
    ax.plot(x1[0], y1[0], z1[0], marker='o', color='k',label='Start')
    ax.plot(0, 0, 0, marker='o', color='r', label='Protostar')
    ax.set_xlim([0,3000])
    ax.set_ylim([0,3000])
    ax.set_zlim([-3000,0])
    # ax.legend()
    return fig,

def animate(i):
    ax.view_init(elev=30., azim=i)
    return fig,

# Animate
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=360, interval=20, blit=True)
# Save
anim.save('test_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
