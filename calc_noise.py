from ChauvenetRMS import *
from astropy.io import fits
import matplotlib.pyplot as plt

image = fits.getdata('C18O/Per-emb-50_C_l025l064_uvsub_C18O_integrated.fits')[0]

rms, sky = calculatenoise(image)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

ax.imshow(sky)
velrange = 5  # km/s
cubenoise = 1.51E-02  # jy/beam
deltav = 0.08534838259220  # km/s
expectednoise = cubenoise * deltav * np.sqrt(velrange/deltav)
print(expectednoise)  # result = 0.009864148395793608 jy/beam km/s
print(rms)  # result = 0.014640286835432989 jy/beam km/s
/data3/dom/MIOP/uvts-RFfix-v2/Per-emb-50/pipe8MHz/l25l64/multiscale/mask2
