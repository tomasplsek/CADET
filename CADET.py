# basic libraries
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from scipy.ndimage import center_of_mass, rotate

# Astropy
from astropy.wcs import WCS
from astropy.io import fits
from astropy.nddata import Cutout2D, CCDData
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel as Gauss

# import ML libraries
from sklearn.cluster import DBSCAN
from tensorflow.keras.models import load_model

# CONFIGURATION FOR DEDICATED NVIDIA GPU 
# from tensorflow.config.experimental import list_physical_devices, set_virtual_device_configuration, VirtualDeviceConfiguration
# gpus = list_physical_devices('GPU')
# set_virtual_device_configuration(gpus[0], [VirtualDeviceConfiguration(memory_limit=1000)])
# print(len(gpus), "Physical GPUs,")

# DISABLES GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

path_model = "CADET.hdf5"
model = load_model(path_model)


def regrid(fname, scale):
    '''
    Crops & regrids image to 128x128.

    This regrid function only handles integer scale factors. 
    If required, for floating point scale factors (0.5, 1.5...), use dmregrid in CIAO.
    '''

    with fits.open(fname) as file:
        wcs0 = WCS(file[0].header)
        data0 = file[0].data
        shape = data0.shape[0]

    # CROP
    x0 = shape / 2
    size = 128 * scale
    cutout = Cutout2D(data0, (x0, x0), (size, size), wcs=wcs0)
    data, wcs = cutout.data, cutout.wcs

    # REGRID DATA
    factor = size // 128
    data = data.reshape(128, factor, 128, factor).mean(-1).mean(1)
    
    # REGIRD WCS
    ra, dec = wcs.wcs_pix2world(np.array([[63, 63]]),0)[0]
    wcs.wcs.cdelt[0] = wcs.wcs.cdelt[0] * factor
    wcs.wcs.cdelt[1] = wcs.wcs.cdelt[1] * factor
    wcs.wcs.crval[0] = ra
    wcs.wcs.crval[1] = dec
    wcs.wcs.crpix[0] = 64 / factor
    wcs.wcs.crpix[1] = 64 / factor

    return data, wcs


def decompose(pred, th2=0.7, amin=10):
    '''
    Decomposes the pixel-wise prediction into individual cavities.
    Applies the higher discrimination threshold and minimal area cut.
    Returns a list of cavities (128x128 matrices).
    '''

    X, Y = pred.nonzero()
    data = np.array([X,Y]).reshape(2, -1)

    # DBSCAN CLUSTERING ALGORITHM
    try: clusters = DBSCAN(eps=1.5, min_samples=3).fit(data.T).labels_
    except: clusters = []

    N = len(set(clusters))
    cavities = []

    for i in range(N):
        img = np.zeros((128,128))
        b = clusters == i
        xi, yi = X[b], Y[b]
        img[xi, yi] = pred[xi, yi]

        # THRESHOLDING #2
        if not (img > th2).any(): continue

        # MINIMAL AREA
        if np.sum(img) <= amin: continue

        cavities.append(img)

    return cavities


def make_cube(image, galaxy, scale, cavity):
    '''
    Assuming rotational symmetry, this function creates a 3D representation of the cavity.
    The 3D cube is saved as a .npy file and can further be used to calculate total cavity energy (E=4pV).
    '''

    # DE-ROTATES CAVITY
    cen = center_of_mass(image)
    phi = np.arctan2(cen[0]-63.5, cen[1]-63.5)
    image = rotate(image, phi*180/np.pi, reshape=False, prefilter=False)
    image = np.where(image > 0.1, 1, 0)

    # ESTIMATES MEANS & WIDTHS IN EACH COLUMN
    means, widths, indices = [], [], []
    for n in range(128):
        rang = np.where(image[:,n] > 0, np.arange(0,128), 0)
        if not (rang > 0).any(): continue
        x = 0
        for i,r in enumerate(rang):
            if r > 0 and x == 0: x = i
            elif x != 0 and r == 0: 
                widths.append(max([(i-x)/2, 0]))
                means.append((x+i)/2)
                indices.append(n)
                x = 0

    # CREATES A 3D CAVITY REPRESENTATION
    cube = np.zeros((128,128,128))
    for m, w, i in zip(means, widths, indices):
        x, y = np.indices((128, 128))
        r = np.sqrt((x-abs(m))**2 + (y-63.5)**2)
        sliced = np.where(r <= w, 1, 0)
        cube[:,:,i] += sliced

    # ROTATES BACK
    cube = rotate(cube, -phi*180/np.pi, axes=(0,2), reshape=False, prefilter=False)
    cube = np.where(cube > 0.1, 1, 0)

    np.save(f"{galaxy}/cubes/{galaxy}_{scale}_{cavity}.npy", cube)


def CADET(galaxy, scales=[1,2,3,4], th1=0.4, th2=0.7):
    galaxy = galaxy.replace(".fits", "")
    # MAKE DIRECTORIES
    os.system(f"mkdir -p {galaxy} {galaxy}/predictions {galaxy}/cubes {galaxy}/decomposed")

    N = len(scales)
    fig, axs = plt.subplots(1, N, figsize=(N*3.2,5))

    for i,scale in enumerate(scales):
        data, wcs = regrid(f"{galaxy}.fits", scale)
        image = np.log10(data+1)

        # ROTATIONAL AVERAGING
        y_pred = 0
        for j in [0,1,2,3]:
            rotated = np.rot90(image, j)
            pred = model.predict(rotated.reshape(1, 128, 128, 1)).reshape(128 ,128)
            pred = np.rot90(pred, -j)
            y_pred += pred / 4

        ccd = CCDData(y_pred, unit="adu", wcs=wcs)
        ccd.write(f"{galaxy}/predictions/{galaxy}_{scale}.fits", overwrite=True)

        # PLOTTING
        ax = axs[i]
        if i == 0:
            ax.text(0.05, 0.95, galaxy, transform=ax.transAxes, color="w", 
                    fontsize=14, va='top', ha='left') #, weight='bold')

        # CONVOLVE IMAGE
        # image = convolve(image, boundary = "extend", nan_treatment="fill",
        #                  kernel = Gauss(x_stddev = 1, y_stddev = 1))

        ax.imshow(image, origin="lower", cmap="viridis", zorder=1) #, norm=LogNorm())

        # PLOT SCALE LINE
        x0, y0 = 0.2, 0.085
        arcsec = 20 * scale
        pix1 = 1 / 128 / 0.492 / scale * arcsec / 2
        pixels = 128 * scale

        # SCALE LINE
        ax.plot([x0-pix1, x0+pix1], [y0, y0], "-", lw=1.3, color="w", transform=ax.transAxes, zorder=3)

        # SCALE IN ARCSEC
        ax.text(x0, y0+0.01, f"{arcsec:.0f} arcsec", va="bottom", ha="center", color="w", transform=ax.transAxes, zorder=3, fontsize=12)

        # SIZE IN PIXELS
        ax.text(x0, y0-0.015, f"{pixels:.0f} pixels", va="top", ha="center", color="w", transform=ax.transAxes, zorder=3, fontsize=12)

        ax.set_xticks([])
        ax.set_yticks([])

        # THRESHOLDING #1
        y_pred = np.where(y_pred > th1, y_pred, 0)

        # CLUSTERING
        cavs = decompose(y_pred, th2, amin=10)

        # PLOT CONTOURS
        if cavs != []: 
            contour = np.array(cavs).sum(axis=0)
            ax.contour(contour, colors=["white","yellow"], linewidths=1.3, levels=[th1, th2], zorder=2, norm=Normalize(0, 1)) #, cmap="viridis")

        for i, cav in enumerate(cavs):
            ax.text(*center_of_mass(cav)[::-1], i+1, color="w", ha="center", va="center", fontsize=14) #, weight="bold")

            ccd = CCDData(cav, unit="adu", wcs=wcs)
            ccd.write(f"{galaxy}/decomposed/{galaxy}_{scale}_{i+1}.fits", overwrite=True)

            make_cube(cav, galaxy, scale, i+1)

    fig.tight_layout()
    fig.savefig(f"{galaxy}/{galaxy}.png", bbox_inches="tight", dpi=200)
    fig.savefig(f"{galaxy}/{galaxy}.pdf", bbox_inches="tight")
    plt.close(fig)


if "__main__" == __name__:
    string = "\nError: Wrong number of arguments.\n"
    string += "Usage: python3 CADET.py galaxy [scales] [threshold1] [threshold2]\n"
    string += "Example: python3 CADET.py NGC4649\n"
    string += "Example: python3 CADET.py NGC4649 [1,2,3,4]\n"
    string += "Example: python3 CADET.py NGC4649 [1,2,3,4] 0.4 0.7\n"
    if len(sys.argv) < 2:
        print(string)

    elif len(sys.argv) == 2:
        CADET(sys.argv[1])

    elif 2 < len(sys.argv) <= 5:
        CADET(sys.argv[1], *[eval(arg) for arg in sys.argv[2:]])

    else:
        print(string)

