# basic libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from IPython.display import display
from scipy.ndimage import center_of_mass, rotate

# Astropy
from astropy.io import fits
from astropy.nddata import Cutout2D, CCDData
from astropy.coordinates import Angle
from astropy.convolution import convolve, Gaussian2DKernel as Gauss
from astropy import units as u
from astropy.wcs import WCS

# import ML libraries
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Filter warnings
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

# Configure GPU
def configure_GPU():
    # # DISABLE GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # CONFIGURATION FOR DEDICATED NVIDIA GPU 
    from tensorflow.config.experimental import list_physical_devices

    # gpus = list_physical_devices('GPU')
    # if len(gpus) > 0:
    #     from tensorflow.config.experimental import set_virtual_device_configuration, VirtualDeviceConfiguration
    #     try: 
    #         set_virtual_device_configuration(gpus[0], [VirtualDeviceConfiguration(memory_limit=1000)])
    #         print(f"\n{len(gpus)} GPUs detected. Configuring the memory limit to 1GB.")
    #     except:
    #         print(f"\n{len(gpus)} GPUs detected. Using default GPU settings.")
    # else: print("\nNo GPUs detected. Using a CPU.")

    gpus = list_physical_devices('GPU')
    if len(gpus) > 0: print(f"{len(gpus)} GPUs detected.\n")
    else: print("No GPUs detected. Using a CPU.\n")

# configure_GPU()
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def rebin(fname, scale, ra="", dec="", shift=False):
    '''
    Crops & rebins image to 128x128 (130x130 if shift=True).
    This rebin function only handles integer scale factors. 
    If required, for floating point scale factors (0.5, 1.5...), use dmregrid in CIAO (https://cxc.cfa.harvard.edu/ciao/).

    Parameters:
    -----------
    fname : str
        Path to the input image.
    scale : int
        Scale factor. The input image will be cropped to 128*scale x 128*scale pixels.
    ra : str, optional
        RA of the center of the image. If not specified, the center of the image will be used.
    dec : str, optional
        DEC of the center of the image. If not specified, the center of the image will be used.
    shift : bool, optional
        If True, the center of the image will be shifted by +/- 1 pixel. This is makes the predictions more robust.

    Returns:
    --------
    data : 2D array
        Re-binned image.
    wcs : WCS object
        WCS of the re-binned image.
    '''

    with fits.open(fname) as file:
        wcs0 = WCS(file[0].header if isinstance(file, fits.HDUList) else file.header)
        data0 = file[0].data if isinstance(file, fits.HDUList) else file.data
        shape = data0.shape

    if (ra != "") and (dec != ""):
        if ":" in ra: ra = Angle(ra, unit="hourangle").degree
        else: ra = float(ra)
        if ":" in dec: dec = Angle(dec, unit="degree").degree
        else: dec = float(dec)
        x0, y0 = wcs0.wcs_world2pix(ra, dec, 0)
    else:
        x0, y0 = shape[0] / 2, shape[1] / 2

    min_size = 130 if shift else 128

    # CROP
    size = min_size * scale
    cutout = Cutout2D(data0, (x0, y0), (size, size), wcs=wcs0)
    data, wcs = cutout.data, cutout.wcs

    # REBIN DATA
    factor = size // min_size
    data = data.reshape(min_size, factor, min_size, factor).sum(-1).sum(1)
    
    # CHANGE WCS
    ra, dec = wcs.wcs_pix2world(np.array([[min_size/2-1, min_size/2-1]]),0)[0]
    wcs.wcs.cdelt[0] = wcs.wcs.cdelt[0] * factor
    wcs.wcs.cdelt[1] = wcs.wcs.cdelt[1] * factor
    wcs.wcs.crval[0] = ra
    wcs.wcs.crval[1] = dec
    wcs.wcs.crpix[0] = min_size / 2 / factor
    wcs.wcs.crpix[1] = min_size / 2 / factor

    return data, wcs


# def bootstrap_counts(data, variance=False):
#     if variance: data = np.random.poisson(data+1)

#     x0, y0 = data.nonzero()
#     probs = data[x0,y0]
#     indices = np.arange(len(x0))

#     sampled_indices = np.random.choice(a=indices, size=int(np.sum(data)), p=probs/np.sum(probs), replace=True)

#     i, v = np.unique(sampled_indices, return_counts=True)

#     sampled_data = np.zeros(data.shape)
#     sampled_data[x0[i],y0[i]] = v

#     return sampled_data


def make_prediction(image, shift=False): #, bootstrap=False, N_bootstrap=10):
    '''
    Apply CADET to an input image.
    Input image must be 128x128 pixels (130x130 if shift=True).
    Shifting makes the prediction more robust by shifting the center of the image by +/- 1 pixel.
    Bootstraping makes the prediction more robust by re-sampling the counts of the input image with replacement.
    Returns pixel-wise prediction (128x128 pixels.

    Parameters:
    -----------
    image : 2D array
        Input image.
    shift : bool, optional
        If True, the center of the image will be shifted by +/- 1 pixel. This is makes the predictions more robust.

    Returns:
    --------
    y_pred : 2D array
        Averaged pixel-wise prediction (128x128 pixels).
    '''

    # Load CADET model
    from keras.models import load_model
    path = os.path.dirname(__file__)
    model = load_model(f'{path}/CADET.hdf5', compile=False, safe_mode=True)

    # N_bootstrap = 1 if not bootstrap else N_bootstrap

    s1, s2 = image.shape
    if s1 != s2: raise ValueError(f"Input image must be square. Current shape: {s1}x{s2}.")
    if (not shift) & (s1 != 128): raise ValueError("Input image must be 128x128 if shift=False.")
    if (shift) & (s1 != 130): raise ValueError("Input image must be 130x130 if shift=True.")

    rotations = [0,1,2,3]
    if shift:
        DX = np.array([0,0,0,1,1,1,-1,-1,-1])
        DY = np.array([0,1,-1,0,1,-1,0,1,-1])
        # DX = np.array([0,0,0,0,0,1,1,1,1,1,-1,-1,-1,-1,-1,2,2,2,-2,-2,-2])
        # DY = np.array([0,1,-1,2,-2,0,1,-1,2,-2,0,1,-1,2,-2,0,1,-1,0,1,-1])
        W = 1 / (1 + DX**2 + DY**2)**0.5
    else:
        DX = np.array([0])
        DY = np.array([0])
        W = np.array([1])

    N_total = len(rotations) * len(DX) # * N_bootstrap

    # NORMALIZE IMAGE
    MIN = np.min(np.where(image == 0, 1, image))
    if MIN < 1: image = image / MIN

    X = []
    for dx, dy, w in zip(DX, DY, W):
        x0 = image.shape[0] // 2
        image_cut = image[x0-dx-64:x0-dx+64, x0-dy-64:x0-dy+64]

        # ROTATIONAL AVERAGING
        for j in rotations:
            rotated0 = np.rot90(image_cut, j)
            # for n in range(N_bootstrap):
            #     if N_bootstrap > 1:
            #         rotated = bootstrap_counts(rotated0)
            #     else: rotated = rotated0

            #     X.append(np.log10(rotated+1))
            
            X.append(np.log10(rotated0+1))

    # APPLY CADET MODEL (runs faster if applied on all images at once, hence the two for loops)
    preds = model.predict(np.array(X).reshape(N_total, 128, 128, 1), verbose=0).reshape(N_total, 128 ,128)

    # SHIFT IMAGES BACK TO ORIGINAL POSITION
    y_pred = np.zeros((N_total, 128, 128))
    x = 0
    for dx, dy, w in zip(DX, DY, W):
        for j in rotations:
            # for n in range(N_bootstrap):
            pred = np.rot90(preds[x], -j)
            prediction = np.zeros((128,128))
            prediction[max(0,dy):128+min(0,dy), max(0,dx):128+min(0,dx)] = pred[max(0,-dy):128+min(0,-dy), max(0,-dx):128+min(0,-dx)]
            y_pred[x,:,:] += prediction * w / sum(W)
            x += 1

    # y_std = np.std(y_pred, axis=0) / len(rotations) / N_bootstrap
    y_pred = np.sum(y_pred, axis=0) / len(rotations) # / N_bootstrap

    return y_pred


def decompose(pred, th1=0.5, th2=0.7, amin=10):
    '''
    Decomposes the pixel-wise prediction into individual cavities.
    Applies the higher discrimination threshold and minimal area cut.
    Returns a list of cavities (128x128 matrices).

    Parameters:
    -----------
    pred : 2D array
        Pixel-wise prediction (128x128 pixels).
    th1 : float, optional
        Volume calibrating threshold (float between 0 and 1).
    th2 : float, optional
        TP/FP calibrating threshold (float between 0 and 1).
    amin : int, optional
        Minimal area of a cavity in pixels.

    Returns:
    --------
    cavities : array
        array of cavity predictions (128x128 matrices).
    '''

    pred = np.where(pred > th1, pred, 0)

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

    return np.array(cavities)


def rotangle_and_ellip(cavity):
    """
    Calculate the rotation angle and ellipticity of a cavity by performing 
    Principal Component Analysis (PCA) using Singular Value Decomposition (SVD).

    Parameters:
    -----------
    cavity: numpy array
        The cavity represented as a numpy array.

    Returns:
    --------
    angle: float
        The rotation angle of the cavity.
    e: float
        The ellipticity of the cavity.
    """

    points = np.array(cavity.nonzero()).T
    pca = PCA(n_components=2)
    pca.fit(points)
    pc1, pc2 = pca.components_
    
    angle = np.arctan2(pc1[1], pc1[0])
    r = pca.singular_values_
    e = (max(r)-min(r))/max(r)
    
    return angle, e


def make_3D_cavity(cavity, rotate_back=False):
    '''
    Assuming rotational symmetry, this function creates a 3D representation of the cavity.
    The 3D cube can be saved as a .npy file and further be used to calculate total cavity energy (E=4pV).
    
    Parameters:
    -----------
    cavity : 2D array
        Pixel-wise prediction of a single cavity (128x128 pixels).

    Returns:
    --------
    cube : 3D array
        3D representation of the cavity (128x128x128).
    '''

    # DE-ROTATES CAVITY
    c0, size = cavity.shape[0]//2 - 0.5, cavity.shape[0]
    cen = center_of_mass(cavity)
    phi = np.arctan2(cen[0]-c0, cen[1]-c0)
    cavity = rotate(cavity, phi*180/np.pi, reshape=False, prefilter=True)
    cavity = np.where(cavity > 0.1, 1, 0)

    # ESTIMATES MEANS & WIDTHS IN EACH COLUMN
    means, widths, indices = [], [], []
    for n in range(size):
        rang = np.where(cavity[:,n] > 0, np.arange(0,size), 0)
        if not (rang > 0).any(): continue
        x = 0
        for i,r in enumerate(rang):
            if r > 0 and x == 0: x = i
            elif x != 0 and r == 0: 
                widths.append(max([(i-x)/2-1, 0]))
                means.append((x+i)/2)
                indices.append(n)
                x = 0

    # CREATES A 3D CAVITY REPRESENTATION
    cube = np.zeros((size,size,size))
    for m, w, i in zip(means, widths, indices):
        x, y = np.indices((size, size))
        r = np.sqrt((x-abs(m))**2 + (y-c0)**2)
        sliced = np.where(r <= w, 1, 0)
        cube[:,:,i] += sliced

    # ROTATES BACK
    if rotate_back:
        cube = rotate(cube, -phi*180/np.pi, axes=(0,2), reshape=False, prefilter=True)
        cube = np.where(cube > 0.1, 1, 0)

    return cube


def CADET(galaxy, scales=[1,2,3,4], ra="", dec="", th1=0.4, th2=0.7, shift=False, plot_smooth=False, plot_arrows=False, verbose=1): #, N_bootstrap=1, bootstrap=False):
    '''
    CADET automated script.

    Parameters:
    -----------
    galaxy : str
        Path to the input image.
    scales : list, optional
        List of scale factors. The input image will be cropped to 128*scale x 128*scale pixels.
    ra : str, optional
        RA of the center of the image. If not specified, the center of the image will be used.
    dec : str, optional
        DEC of the center of the image. If not specified, the center of the image will be used.
    th1 : float, optional
        Volume calibrating threshold (float between 0 and 1).
    th2 : float, optional
        TP/FP calibrating threshold (float between 0 and 1).
    shift : bool, optional
        If True, the center of the image will be shifted by +/- 1 pixel. This is makes the predictions more robust.
    plot_smooth : bool, optional
        If True, the input image will be smoothed before plotting.
    plot_arrows : bool, optional
        If True, the major and minor axis of the cavity will be plotted as arrows.
    verbose : int, optional
        If 0, no output will be printed. If 1, the output will be printed.

    Returns:
    --------
    Creates the following files:
    - {galaxy}/predictions/{galaxy}_{scale}.fits - raw CADET predictions
    - {galaxy}/decomposed/{galaxy}_{scale}_{i+1}.fits - predictions decomposed into individual cavities
    - {galaxy}/cubes/{galaxy}_{scale}_{i+1}.npy - 3D representations of cavities
    - {galaxy}/{galaxy}.png - plot of the input image with detected cavities
    '''

    if verbose: 
        print("\033[92m---- Running CADET ----\033[0m")

        print(f"Reading file: {galaxy}")
    galaxy = galaxy.replace(".fits", "")

    # Load image
    hdu0 = fits.open(f"{galaxy}.fits")
    image0 = hdu0[0].data
    wcs0 = WCS(hdu0[0].header)
    if verbose:
        print(f"\nOriginal image size: {image0.shape[0]} x {image0.shape[1]} pixels")
        print(f"Selected scales: {str(scales)}")

    # Print RA & DEC, if not specified use the center of the image
    if (ra != "") and (dec != ""):
        if verbose:
            print(f"RA:  {ra} hours")
            print(f"DEC: {dec} degrees")
    else:
        RA, DEC = wcs0.wcs_pix2world(image0.shape[0]/2, image0.shape[1]/2, 0)
        RA = Angle(RA, unit="degree").to_string(unit=u.hour, sep=':', precision=2)
        DEC = Angle(DEC, unit="degree").to_string(unit=u.degree, sep=':', precision=2)
        if verbose:
            print("\nRA & DEC not specified.\nUsing the center of the image:")
            print(f"RA:  {RA} hours")
            print(f"DEC: {DEC} degrees")

    # MAKE DIRECTORIES
    if verbose:
        # print(f"Creating directories {galaxy}:\n{galaxy}/\n  \u251Cpredicitons/ - raw CADET predictions\n  \u251Cdecomposed/ - predictions decomposed into individual cavities\n  \u2514cubes/ - 3D representations of cavities")
        print(f"\nCreating directories:\n{galaxy}/\n  \u251C cropped/\n  \u251C predicitons/\n  \u251C decomposed/\n  \u2514 cubes/")
    os.system(f"mkdir -p {galaxy} {galaxy}/predictions {galaxy}/decomposed {galaxy}/cubes {galaxy}/cropped")

    # Blank dataframe for saving results
    index = pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=['scale', 'cavity'])
    df = pd.DataFrame(index=index)

    # Create matplotlib figure
    N = len(scales)
    fig, axs = plt.subplots(1, N, figsize=(N*3.2,5))

    if verbose: print("\nProcessing the image on following scales:")
    for i,scale in enumerate(scales):
        size = 128 * scale
        if verbose:
            print(f"{size} pixels:", end="  ")

        image, wcs = rebin(f"{galaxy}.fits", scale, ra=ra, dec=dec, shift=shift)
        hdu = fits.PrimaryHDU(image, header=wcs.to_header())
        hdu.writeto(f"{galaxy}/cropped/{galaxy}_{scale * 128}.fits", overwrite=True)

        angular_scale = wcs.pixel_scale_matrix[1,1] * 3600

        y_pred = make_prediction(image, shift=shift) #, bootstrap=bootstrap, N_bootstrap=N_bootstrap)

        ccd = CCDData(y_pred, unit="adu", wcs=wcs)
        ccd.write(f"{galaxy}/predictions/{galaxy}_{size}.fits", overwrite=True)

        # PLOTTING
        ax = axs[i]
        if i == 0:
            ax.text(0.05, 0.95, galaxy, transform=ax.transAxes, color="w", 
                    fontsize=14, va='top', ha='left') #, weight='bold')

        # CONVOLVE IMAGE
        if plot_smooth:
            image = convolve(np.log10(image+1), boundary = "extend", nan_treatment="fill",
                            kernel = Gauss(x_stddev = 1, y_stddev = 1))
        else:
            image = np.log10(image+1)

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

        # CLUSTERING
        cavs = decompose(y_pred, th1, th2, amin=12)

        if verbose:
            print(f"detected {len(cavs)} {'cavity' if len(cavs) == 1 else 'cavities'}")

        # PLOT CONTOURS
        if cavs.size > 0: 
            ax.contour(cavs.sum(axis=0), colors=["white","yellow"], linewidths=1.3, levels=[th1, th2], zorder=2, norm=Normalize(0,1))

        for i, cav in enumerate(cavs):
            center = center_of_mass(cav.T)
            ha, va = "center", "center"
            ax.text(*center, i+1, color="w", ha=ha, va=va, fontsize=14, weight="bold")

            angle, e = rotangle_and_ellip(cav.T)

            if plot_arrows:
                a = (np.sum(cav) / np.pi / (1-e))**0.5
                b = a * (1-e)

                ax.quiver(center[0], center[1], np.cos(angle)*a, np.sin(angle)*a, 
                        color='cyan', scale=1, scale_units='xy', angles='xy', zorder=1)
                ax.quiver(center[0], center[1], -np.sin(angle)*b, np.cos(angle)*b, 
                        color='white', scale=1, scale_units='xy', angles='xy', zorder=1)

            ccd = CCDData(cav, unit="adu", wcs=wcs)
            ccd.write(f"{galaxy}/decomposed/{galaxy}_{size}_{i+1}.fits", overwrite=True)

            cube = make_3D_cavity(cav, rotate_back=False)
            np.save(f"{galaxy}/cubes/{galaxy}_{size}_{i+1}.npy", cube)

            area = np.sum(cav)
            area_arcsec = area * angular_scale**2
            volume = np.sum(cube)
            volume_arcsec = volume * angular_scale**3

            df.loc[(f"{size} pixels", i+1), "area [px²]"] = round(area)
            df.loc[(f"{size} pixels", i+1), "area [arcsec²]"] = round(area_arcsec)
            df.loc[(f"{size} pixels", i+1), "volume [px³]"] = round(volume)
            df.loc[(f"{size} pixels", i+1), "volume [arcsec³]"] = round(volume_arcsec)
            df.loc[(f"{size} pixels", i+1), "angle [deg]"] = round(np.degrees(angle), 1)
            df.loc[(f"{size} pixels", i+1), "ellipticity"] = round(e, 2)

    # Save & display results
    if verbose:
        print(f"\nSaving results:\n{galaxy}/cavity_properties.txt")
        print("\narea [px²] and volume [px³] are expressed in units of binned pixels")
        print("volume [arcsec³] - calculated assuming rotational symmetry along the axis from galaxy center to cavity center")

    df.to_csv(f"{galaxy}/cavity_properties.txt", sep=",", float_format="%.2f")
    if verbose:
        df["area [px²]"] = df["area [px²]"].astype(int)
        df["area [arcsec²]"] = df["area [arcsec²]"].astype(int)
        df["volume [px³]"] = df["volume [px³]"].astype(int)
        df["volume [arcsec³]"] = df["volume [arcsec³]"].astype(int)
        display(df)

    fig.tight_layout()
    fig.savefig(f"{galaxy}/{galaxy}.png", bbox_inches="tight", dpi=250)
    # plt.close(fig)

