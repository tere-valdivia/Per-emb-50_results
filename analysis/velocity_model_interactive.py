import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import pickle
from scipy import stats
import astropy.units as u
import velocity_tools.stream_lines as SL
from astropy.wcs import WCS
from astropy.io import fits
from NOEMAsetup import *
from astropy.coordinates import SkyCoord, FK5
import pyregion
import pickle

# Main parameters to generate a streamline
# Mstar = 0.58*u.Msun
Mstar = (2.9+2.2+0.58)*u.Msun # mass of the star and envelope and disk
# inc = -(67-180)*u.deg
inc = (360-(90-67))*u.deg # should be almost edge on
# inc = (360-(90-77))*u.deg # should be almost edge on
PA_ang = -(170-90)*u.deg
regionsample = 'data/region_streamer_l.reg'

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
