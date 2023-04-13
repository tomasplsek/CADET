# Basic libraries
import os, glob, sys
import numpy as np
import pandas as pd

# Scipy
from scipy.ndimage import center_of_mass
from scipy.special import beta as betafun

# Astropy
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, ListedColormap, Normalize
from matplotlib.ticker import FuncFormatter, ScalarFormatter
fsize, fsize2 = 16, 19
plt.rc('font', size=fsize)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{newtxtext}\usepackage{newtxmath}')

import warnings
warnings.filterwarnings('ignore') # :-)

# Tensorflow & JAX library
from jax.numpy import stack, asarray
from tensorflow.keras.models import load_model
from tensorflow.config.experimental import list_physical_devices, set_virtual_device_configuration, VirtualDeviceConfiguration

gpus = list_physical_devices('GPU')
set_virtual_device_configuration(gpus[0], [VirtualDeviceConfiguration(memory_limit=2000)])
print(len(gpus), "Physical GPUs")
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.65'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# custom libraries
sys.path.append("/home/plsek/Diplomka/CADET") 
from beta_model import get_batch, run_vect
from functions import *


q1, q3 = 0.15865525, 0.84134475

path_model = "best.hdf5"
# path_model = "CADET_final.hdf5"
# path_model = "/home/plsek/Diplomka/CADET/models/CADET_size.hdf5"
# path_model = "models/b16_lr0.0005_normal_rims_nosloshing_flatter_50_customunet.hdf5"
model = load_model(path_model)

parnames = ["dx", "dy", "dx_2", "dy_2", "phi", "phi_2",
            "A", "r0", "beta", "ellip", "A_2", "r0_2", "beta_2", "bkg", 
            "s_depth", "s_period", "s_dir", "s_angle",
            "r1", "r2", "phi1", "phi2", "theta1", "theta2", "R1", "R2", "e1", "e2", "varphi1", "varphi2",
            "rim_size", "rim_height", "rim_type"]


def augment(Xs):
    Xrot1 = np.array([np.rot90(X, k=1) for X in Xs])
    Xrot2 = np.array([np.rot90(X, k=2) for X in Xs])
    Xrot3 = np.array([np.rot90(X, k=3) for X in Xs])
    return np.concatenate((Xs, Xrot1, Xrot2, Xrot3))


def estimate_error(model, Xs, ys, vs, threshold, threshold2, FPrate=False):
    FP, TP = 0, 0
    distances, radii = [], []
    pred_A, pred_V = [], []

    Xs = np.log10(Xs+1)
    y_pred = model.predict(Xs.reshape(-1, 128, 128, 1))
    y_pred = y_pred.reshape(-1, 128, 128)

    for i in range(len(y_pred)):
        pred = np.where(y_pred[i] > threshold, y_pred[i], 0)
        cavs = decompose_two(pred, threshold2)
        
        tp, fp, img, cube = 0, 0, 0, 0
        for i1, cav in enumerate(cavs):
            if FPrate & (cav>0).any(): 
                fp += 0.5
                distances.append(np.sqrt(sum((center_of_mass(cav)-np.array([64,64]))**2)))
                radii.append(np.sqrt(np.sum(cav)/(np.pi)))
            # if ((cav>0) & (ys[i]>0)).any():
            if np.sum((cav>0) & (ys[i]>0)) >= 0.20 * np.sum(ys[i]):
                tp += 0.5
                img += cav
                cube += make_cube(cav)
            else: fp += 0.5

        if tp == 1:
            pred_A.append((np.sum(img) / np.sum(ys[i]) - 1)*100)
            pred_V.append((np.sum(cube) / vs[i] - 1)*100)

        TP += tp
        FP += fp

    if FPrate: return FP, np.array(distances), np.array(radii)
    else: return FP, TP, np.array(pred_A), np.array(pred_V)


def relative_error(galaxy, cavity, scale, N=1000, N2=40, threshold=0.5, threshold2=0.7,
                   x0_1=0, y0_1=0, x0_2=0, y0_2=0,
                   ampl=20, r0=10, alpha=1, bkg=0,
                   ampl_2=0, r0_2=0, alpha_2=0,
                   ellip=0, phi_1=0, phi_2=0,
                   r1=0, r2=0, R1=0, R2=0,
                   varphi1=0, varphi2=0, fac=1):
    
    # LOAD REAL IMAGE
    real = fits.getdata(glob.glob(f"real_data/All/{galaxy}_{int(scale*128)}.fits")[0])

    # PRIMARY PARAMETERS
    A = (ampl / r0 / betafun(alpha, 0.5))**0.5
    beta = (alpha + 0.5) / 3

    # SECONDARY PARAMETERS
    A_2 = np.where(ampl_2 > 0, (ampl_2 / r0_2 / betafun(alpha_2, 0.5))**0.5, 0)
    beta_2 = np.where(ampl_2 > 0, (alpha_2 + 0.5) / 3, 0)

    dx, dy, dx_2, dy_2 = 0, 0, 0, 0
    s_depth, s_period, s_dir, s_angle = 0, 0, 0, 0
    rim_size, rim_height, rim_type = 0, 0, 0

    theta1, theta2, phi1, phi2 = 0, 0, 0, 0
    e1, e2 = 0, 0

    # e1 = stack(exponential(0.15, 0.6, N))
    # e2 = stack(exponential(0.15, 0.6, N))

    theta1 = np.zeros(N) # stack(np.random.normal(0, 25, size=N))
    theta2 = np.zeros(N) # -theta1 + np.random.normal(0, 10, size=N)

    s1 = lambda x: stack([float(x)]*N2)
    s2 = lambda x,i,N2: stack(list(x)[i*N2:(i+1)*N2])

    # TRUE-POSITIVE RATE, VOLUME ERROR
    # R1, R2, r1, r2, varphi1, varphi2 = 15, 15, 35, 35, 0, 180
    Xs, ys, vs = [], [], []
    for i in range(N//N2):
        n = stack([i for i in range(i*N2,(i+1)*N2)])
        X, y, v = run_vect(n, s1(dx), s1(dy), s1(dx_2), s1(dy_2), s2(phi_1,i,N2), s2(phi_2,i,N2),
                            s2(A,i,N2), s2(r0,i,N2), s2(beta,i,N2), s2(ellip,i,N2),
                            s2(A_2,i,N2), s2(r0_2,i,N2), s2(beta_2,i,N2), s2(bkg,i,N2), 
                            s1(s_depth), s1(s_period), s1(s_dir), s1(s_angle),
                            s1(r1), s1(r2), s1(varphi1), s1(varphi2), s2(theta1,i,N2), s2(theta2,i,N2), 
                            s1(R1), s1(R2), s1(e1), s1(e2), s1(phi1), s1(phi2),
                            s1(rim_size), s1(rim_height), s1(rim_type))
        try: Xs, ys, vs = np.concatenate((Xs, X)), np.concatenate((ys, y)), np.concatenate((vs, v))
        except: Xs, ys, vs = X, y, v

    # PLOT EXEMPLAR IMAGES
    for i in range(10):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        ax1.imshow(real+1, norm=LogNorm(), origin="lower")
        ax1.axis("off")        
        ax2.imshow(Xs[i]+1, norm=LogNorm(), origin="lower")
        ax2.axis("off")
        fig.savefig(f"real_data/simulated/{galaxy}_{i}.png", bbox_inches="tight")
        plt.close(fig)

    # SET THRESHOLDS BASED ON NUMBER OF COUNTS
    counts, counts_std = np.mean(np.sum(Xs, axis=(1,2))), np.std(np.sum(Xs, axis=(1,2)))
    if counts < 2e4: threshold, threshold2 = 0.32, 0.6
    elif counts < 2e5: threshold, threshold2 = 0.35, 0.8
    else: threshold, threshold2 = 0.38, 0.85

    # AUGMENT & ESTIMATE ERROR
    # Xs, ys, vs = augment(Xs, ys, vs)
    FP, TP, area, volume = estimate_error(model, Xs, ys, vs, threshold, threshold2, FPrate=False)

    # FALSE-POSITIVE RATE
    # R1, R2, r1, r2 = 0, 0, 0, 0
    Xs, ys, vs = [], [], []
    for i in range(N//N2):
        n = stack([i for i in range(i*N2,(i+1)*N2)])
        X, y, v = run_vect(n, s1(dx), s1(dy), s1(dx_2), s1(dy_2), s2(phi_1,i,N2), s2(phi_2,i,N2),
                            s2(A,i,N2), s2(r0,i,N2), s2(beta,i,N2), s2(ellip,i,N2),
                            s2(A_2,i,N2), s2(r0_2,i,N2), s2(beta_2,i,N2), s2(bkg,i,N2), 
                            s1(s_depth), s1(s_period), s1(s_dir), s1(s_angle),
                            s1(0), s1(0), s1(varphi1), s1(varphi2), s2(theta1,i,N2), s2(theta2,i,N2), 
                            s1(0), s1(0), s1(e1), s1(e2), s1(phi1), s1(phi2),
                            s1(rim_size), s1(rim_height), s1(rim_type))
        try: Xs, ys, vs = np.concatenate((Xs, X)), np.concatenate((ys, y)), np.concatenate((vs, v))
        except: Xs, ys, vs = X, y, v

    # AUGMENT & ESTIMATE FP-RATE
    # Xs, ys, vs = augment(Xs, ys, vs)
    FP_nocav, distances, radii = estimate_error(model, Xs, ys, vs, threshold, threshold2, FPrate=True)

    # CALCULATE TP & FP RATES, VOLUME ERRORS
    b = ((r1 - R1) < distances) & (distances < (r1 + R1)) & (R1/2 < radii)
    TP = TP / len(Xs)
    FP = FP / len(Xs)
    FP_nocav = FP_nocav / len(Xs)
    FP_nocav_similar = np.sum(b) / len(Xs) / 2 if len(b) > 0 else 0
    area = np.array([0]) if len(area) == 0 else area
    volume = np.array([0]) if len(volume) == 0 else volume

    # PRINT TP & FP RATES, VOLUME ERRORS
    print_error = lambda x: "{0:.1f} +{2:.1f} {1:.1f}".format(np.median(x), *(np.quantile(x, (0.25,0.75))-np.median(x)))
    print_error2 = lambda x: "${0:.0f}^{{+{2:.0f}}}_{{{1:.0f}}}$".format(np.median(x), *(np.quantile(x, (0.25,0.75))-np.median(x)))
    print()
    print(galaxy)
    print(R1, r1)
    print("area error:", print_error(area))
    print("abs area error:", print_error(abs(area)))
    print("volume error:", print_error(volume))
    print("abs volume error:", print_error(abs(volume)))
    print(f"TP: {TP:.3f}")
    print(f"FP: {FP:.3f}")
    print(f"FP_nocav: {FP_nocav:.3f}")
    print(f"FP_nocav_similar: {FP_nocav_similar:.3f}")

    # PLOT RADIAL PROFILES
    profs = []
    for i in range(len(Xs)):
        profs.append(get_radial_profile(Xs[i])[1]) #, center=(63.5, 63.5))[1])

    # LOAD REAL IMAGE
    x, y = get_radial_profile(real) #, center=(63.5, 63.5))
    Ymed = np.median(profs, axis=0)
    Y16,Y1,Y3,Y84 = np.quantile(np.array(profs)/fac, (0.1, 0.25, 0.75, 0.99), axis=0)

    # FIGURE
    fontsize, labelpad, linewidth = 14, 8, 1.3
    fig, ax = plt.subplots(figsize=(6, 6))

    plt.setp(ax.spines.values(), linewidth=linewidth)
    ax.tick_params(axis="both", which="major", length=8, width=linewidth, labelsize=fontsize)
    ax.tick_params(axis="both", which="minor", length=4, width=1.1)

    # MAKE TITLE
    r = lambda x: round(x, -2)
    ax.set_title(f"{galaxy}\ncounts: {r(np.sum(real)):.0f}, sim counts: {r(counts):.0f} +/- {r(counts_std):.0f}")

    # PLOT SIMULATED PROFILE
    ax.fill_between(x[:len(Y1)], Y16, Y84, color="k", alpha=0.3)
    ax.fill_between(x[:len(Y1)], Y1, Y3, color="k", alpha=0.4)
    ax.plot(x[:len(Y1)], Ymed, lw=1.5, ls="-", ms=0, c="k")

    # PLOT REAL PROFILE
    ax.plot(x, y)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ylim = ax.get_ylim()
    ax.set_ylim(bottom=max(1e-2, ylim[0]))
    formatter = FuncFormatter(lambda y, _: "10$^{{\\text{{{:.0f}}}}}$".format(np.log10(y)))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel("radius (pixels)", fontsize=fontsize, labelpad=labelpad)
    ax.set_ylabel("surface brightness (counts$\,$/$\,$pixel$^{\\text{2}}$)", fontsize=fontsize, labelpad=labelpad)

    pref = "single_" if A_2[0] == 0 else "double_"
    fig.savefig(f"real_data/profiles/{pref}{galaxy}.png", bbox_inches="tight")

    return round(FP_nocav_similar, 2), round(TP, 2), print_error2(volume)


def get_radial_profile(data, center=None):
    if not center: center = np.array(data.shape) / 2
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)
    
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    R = np.linspace(np.min(r), np.max(r), len(radialprofile))
    return R, radialprofile


def angle_to_side(angle, i=""):
	if angle >= 337.5 or angle < 22.5: return "W"+str(i)
	elif 22.5 <= angle < 67.5: return "NW"+str(i)
	elif 67.5 <= angle < 112.5: return "N"+str(i)
	elif 112.5 <= angle < 157.5: return "NE"+str(i)
	elif 157.5 <= angle < 202.5: return "E"+str(i)
	elif 202.5 <= angle < 247.5: return "SE"+str(i)
	elif 247.5 <= angle < 292.5: return "S"+str(i)
	elif 292.5 <= angle < 337.5: return "SW"+str(i)


########################## BETA MODELS ##########################

df_beta = pd.read_csv("beta_models.csv", index_col=0)

df_beta_ellip = pd.read_csv("beta_models_ellip.csv", index_col=0)

for par in [".ellip", ".ellip+", ".ellip-", ".theta", ".theta+", ".theta-"]:
    for i in range(1,2):
        df_beta[f"b{i}{par}"] = df_beta_ellip[f"b{i}{par}"] 

galaxies = pd.read_csv("Galaxies.csv")
gals = galaxies["Galaxy"][galaxies["Sample_C22"] == 1]
galaxies.index = galaxies["Galaxy"]
D_SBF = galaxies["D_SBF"]

df_beta["Size"] = df_beta["size"] // 128

############################ CAVITIES ############################

cavities = pd.read_csv("cavities_new.csv", index_col=0)
fac = 0.492

def run_heatmap(galaxy):
    df = df_beta.loc[galaxy]

    N, N2 = 1000, 40
    fac = 1

    x0_1 = np.random.normal(df["b1.xpos"], df["b1.xpos+"], size=N)
    y0_1 = np.random.normal(df["b1.ypos"], df["b1.ypos+"], size=N)
    ampl = np.random.normal(df["b1.ampl"], df["b1.ampl+"], size=N) * fac
    r0 = np.random.normal(df["b1.r0"], df["b1.r0+"], size=N)
    alpha = np.random.normal(df["b1.alpha"], df["b1.alpha+"], size=N)
    ellip = np.random.normal(df["b1.ellip"], df["b1.ellip+"], size=N)
    phi_1 = np.random.normal(df["b1.theta"], df["b1.theta+"], size=N)
    bkg = np.random.normal(df["bkg.c0"], df["bkg.c0+"], size=N) * fac

    x0_2 = np.random.normal(df["b2.xpos"], df["b2.xpos+"], size=N)
    y0_2 = np.random.normal(df["b2.ypos"], df["b2.ypos+"], size=N)
    ampl_2 = np.random.normal(df["b2.ampl"], df["b2.ampl+"], size=N) * fac
    r0_2 = np.random.normal(df["b2.r0"], df["b2.r0+"], size=N)
    alpha_2 = np.random.normal(df["b2.alpha"], df["b2.alpha+"], size=N)
    phi_2 = np.random.normal(df["b2.theta"], df["b2.theta+"], size=N)

    r1, r2, R1, R2, varphi1, varphi2 = 0, 0, 0, 0, 0, 0

    files = glob.glob(f"real_data/Detected/{galaxy}_*_*.fits")
    scales = [float(fname.split("_")[-2]) for fname in files]
    for fname in files:
        cavity = int(fname.split("_")[-1].split(".")[0])
        scale = float(fname.split("_")[-2])
        df.Size = scale

        file = fits.open(fname)
        data = file[0].data
        s = data.shape[0] // 2 - 0.5
        wcs = WCS(file[0].header)
        x, y = center_of_mass(data)
        r = np.sqrt((x - 63.5)**2 + (y - 63.5)**2)
        R = np.sqrt(np.sum(data)/np.pi)

        center = SkyCoord(*wcs.wcs_pix2world([[63, 63]], 0)[0], unit="deg", frame="fk5")
        cavity = SkyCoord(*wcs.wcs_pix2world([[y, x]], 0)[0], unit="deg", frame="fk5")
        varphi = (center.position_angle(cavity).deg + 90) % 360

        gen = 2 if (len(scales) > 2) & (scale == max(scales)) else 1
        side = angle_to_side(varphi, gen)

        FP, TP, error = relative_error(galaxy, cavity, scale, N=N, N2=N2, threshold=0.3, threshold2=0.65,
                        x0_1=x0_1, y0_1=y0_1, x0_2=x0_2, y0_2=y0_2,
                        ellip=ellip, phi_1=phi_1, phi_2=phi_2,
                        ampl=ampl*df.Size**2, r0=r0/df.Size, alpha=alpha, bkg=bkg*df.Size**2,
                        ampl_2=ampl_2*df.Size**2, r0_2=r0_2/df.Size, alpha_2=alpha_2,
                        r1=r, r2=r, R1=R, R2=R, varphi1=varphi, varphi2=varphi+180, fac=fac)

        print(varphi, side)

        # WRITE RESULTS TO CSV
        significances_fname = "real_data/significances.csv"
        df = pd.read_csv(significances_fname, index_col=[0,1])
        df.loc[(galaxy, side), ["FP", "TP", "Error"]] = [FP, TP, error]
        df.to_csv(significances_fname)


gals = df_beta.index #[df_beta["model"] == "Single beta"]
new_cavities = ["IC1860", "IC4765", "NGC499", "NGC533", "NGC720", "NGC1399", "NGC1521", "NGC1700", "NGC2300", "NGC3091", "NGC3402", "NGC3923", "NGC4073", "NGC4104", "NGC4125", "NGC4325", "NGC4472", "NGC4526", "NGC4555", "NGC5129", "NGC6482"]

if len(sys.argv) == 1: 
    for galaxy in new_cavities: run_heatmap(galaxy)
else: run_heatmap(sys.argv[1])
