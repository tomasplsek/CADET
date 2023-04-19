# Basic libraries
import os
import shutil
import numpy as np
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

# Astropy
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D, CCDData
from astropy.convolution import Gaussian2DKernel as Gauss
from astropy.convolution import convolve

# Scikit-learn
from sklearn.cluster import DBSCAN

# Streamlit
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Cavity Detection Tool", layout="wide")

# HuggingFace Hub
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from huggingface_hub import from_pretrained_keras
# from tensorflow.keras.models import load_model


# Define function to plot the uploaded image
def plot_image(image, scale):
    plt.figure(figsize=(4, 4))
    x0 = image.shape[0] // 2 - scale * 128 / 2
    plt.imshow(image, origin="lower")
    plt.gca().add_patch(Rectangle((x0-0.5, x0-0.5), scale*128, scale*128, linewidth=1, edgecolor='w', facecolor='none'))
    plt.axis('off')
    plt.tight_layout()
    with colA: st.pyplot()

# Define function to plot the prediction
def plot_prediction(pred):
    plt.figure(figsize=(4, 4))
    plt.imshow(pred, origin="lower", norm=Normalize(vmin=0, vmax=1))
    plt.axis('off')
    with colB: st.pyplot()

# Define function to plot the decomposed prediction
def plot_decomposed(decomposed):
    plt.figure(figsize=(4, 4))
    plt.imshow(decomposed, origin="lower")

    N = int(np.max(decomposed))
    for i in range(N):
        new = np.where(decomposed == i+1, 1, 0)
        x0, y0 = center_of_mass(new)
        color = "white" if i < N//2 else "black"
        plt.text(y0, x0, f"{i+1}", ha="center", va="center", fontsize=15, color=color)
    
    plt.axis('off')
    with colC: st.pyplot()
        
# Define function to cut input image and rebin it to 128x128 pixels
def cut(data0, wcs0, scale=1):
    shape = data0.shape[0]
    x0 = shape / 2
    size = 128 * scale
    cutout = Cutout2D(data0, (x0, x0), (size, size), wcs=wcs0)
    data, wcs = cutout.data, cutout.wcs

    # Regrid data
    factor = size // 128
    data = data.reshape(128, factor, 128, factor).mean(-1).mean(1)
    
    # Regrid wcs
    ra, dec = wcs.wcs_pix2world(np.array([[63, 63]]),0)[0]
    wcs.wcs.cdelt[0] = wcs.wcs.cdelt[0] * factor
    wcs.wcs.cdelt[1] = wcs.wcs.cdelt[1] * factor
    wcs.wcs.crval[0] = ra
    wcs.wcs.crval[1] = dec
    wcs.wcs.crpix[0] = 64 / factor
    wcs.wcs.crpix[1] = 64 / factor

    return data, wcs

# Define function to apply cutting and produce a prediction
@st.cache_data
def cut_n_predict(data, _wcs, scale):
    data, wcs = cut(data, _wcs, scale=scale)
    image = np.log10(data+1)
    
    y_pred = 0
    for j in [0,1,2,3]:
        rotated = np.rot90(image, j)
        pred = model.predict(rotated.reshape(1, 128, 128, 1)).reshape(128 ,128)
        pred = np.rot90(pred, -j)
        y_pred += pred / 4

    return y_pred, wcs

# Define function to decompose prediction into individual cavities
@st.cache_data
def decompose_cavity(pred, fname, th2=0.7, amin=10):
    X, Y = pred.nonzero()
    data = np.array([X,Y]).reshape(2, -1)

    # DBSCAN clustering
    try: clusters = DBSCAN(eps=1.0, min_samples=3).fit(data.T).labels_
    except: clusters = []

    N = len(set(clusters))
    cavities = []

    for i in range(N):
        img = np.zeros((128,128))
        b = clusters == i
        xi, yi = X[b], Y[b]
        img[xi, yi] = pred[xi, yi]

        # # Thresholding #2
        # if not (img > th2).any(): continue

        # Minimal area
        if np.sum(img) <= amin: continue

        cavities.append(img)

    # Save raw and decomposed predictions to predictions folder
    ccd = CCDData(pred, unit="adu", wcs=wcs)
    ccd.write(f"{fname}/predicted.fits", overwrite=True)
    image_decomposed = np.zeros((128,128))
    for i, cav in enumerate(cavities):
        ccd = CCDData(cav, unit="adu", wcs=wcs)
        ccd.write(f"{fname}/decomposed_{i+1}.fits", overwrite=True)
        image_decomposed += (i+1) * np.where(cav > 0, 1, 0)

    # shutil.make_archive("predictions", 'zip', "predictions")
    
    return image_decomposed

@st.cache_data
def load_file(fname):
    with fits.open(fname) as hdul:
        data = hdul[0].data
        wcs = WCS(hdul[0].header)
    return data, wcs

@st.cache_resource
def load_CADET():
    model = from_pretrained_keras("Plsek/CADET-v1")
    # model = load_model("CADET.hdf5")
    return model

def reset_threshold():
    del st.session_state["threshold"]


# Load model
model = load_CADET()

# Use wide layout and create columns
bordersize = 0.6
_, col, _ = st.columns([bordersize, 3, bordersize])

os.system("rm *.zip")
os.system("rm -R -- */")
# if os.path.exists("predictions"): os.system("rm -r predictions")
# os.system("mkdir -p predictions")

with col:
    # Create heading and description
    st.markdown("<h1 align='center'>Cavity Detection Tool</h1>", unsafe_allow_html=True)    
    st.markdown("Cavity Detection Tool (CADET) is a machine learning pipeline trained to detect X-ray cavities from noisy Chandra images of early-type galaxies.")
    st.markdown("To use this tool: upload your image, select the scale of interest, make a prediction, and decompose it into individual cavities!")
    st.markdown("Input images should be in units of counts, centred at the galaxy center, and point sources should be filled with surrounding background ([dmfilth](https://cxc.cfa.harvard.edu/ciao/ahelp/dmfilth.html)).")
    st.markdown("If you use this tool for your research, please cite [Plšek et al. 2023](https://arxiv.org/abs/2304.05457)")

# _, col_1, col_2, col_3, _ = st.columns([bordersize, 2.0, 0.5, 0.5, bordersize])

# with col:
    uploaded_file = st.file_uploader("Choose a FITS file", type=['fits']) #, on_change=reset_threshold)

    # with col_2:
    #     st.markdown("### Examples")
    #     NGC4649 = st.button("NGC4649")

    # with col_3:
    #     st.markdown("""<style>[data-baseweb="select"] {margin-top: 26px;}</style>""", unsafe_allow_html=True)
    #     NGC5813 = st.button("NGC5813")

    # if NGC4649:
    #     uploaded_file = "NGC4649_example.fits"
    # elif NGC5813:
    #     uploaded_file = "NGC5813_example.fits"

    # If file is uploaded, read in the data and plot it
if uploaded_file is not None:
    data, wcs = load_file(uploaded_file)
    os.mkdir(uploaded_file.name.strip(".fits"))

if "data" not in locals():
    data = np.zeros((128,128))
        
# Make six columns for buttons
_, col1, col2, col3, col4, col5, col6, _ = st.columns([bordersize,0.5,0.5,0.5,0.5,0.5,0.5,bordersize])
col1.subheader("Input image")
col3.subheader("Prediction")
col5.subheader("Decomposed")
col6.subheader("")

with col1:
    st.markdown("""<style>[data-baseweb="select"] {margin-top: -46px;}</style>""", unsafe_allow_html=True)
    max_scale = int(data.shape[0] // 128)
    scale = st.selectbox('Scale:',[f"{(i+1)*128}x{(i+1)*128}" for i in range(max_scale)], label_visibility="hidden", on_change=reset_threshold)
    scale = int(scale.split("x")[0]) // 128

# Detect button
with col3: detect = st.button('Detect', key="detect")

# Threshold slider
with col4:
    st.markdown("")
    # st.markdown("""<style>[data-baseweb="select"] {margin-top: -36px;}</style>""", unsafe_allow_html=True)
    threshold = st.slider("Threshold", 0.0, 1.0, 0.0, 0.05, key="threshold") #, label_visibility="hidden")
    
# Decompose button
with col5: decompose = st.button('Decompose', key="decompose")
    
# Make two columns for plots
_, colA, colB, colC, _ = st.columns([bordersize,1,1,1,bordersize])

if uploaded_file is not None:
    image = np.log10(data+1)
    plot_image(image, scale)

    if detect or threshold or st.session_state.get("decompose", False):
        fname = uploaded_file.name.strip(".fits")

        y_pred, wcs = cut_n_predict(data, wcs, scale)
        
        y_pred_th = np.where(y_pred > threshold, y_pred, 0)
                
        plot_prediction(y_pred_th)

        if decompose or st.session_state.get("download", False):            
            image_decomposed = decompose_cavity(y_pred_th, fname)

            plot_decomposed(image_decomposed)

            with col6:
                st.markdown("<br style='margin:4px 0'>", unsafe_allow_html=True)
                # st.markdown("""<style>[data-baseweb="select"] {margin-top: 16px;}</style>""", unsafe_allow_html=True)
        
                # if st.session_state.get("download", False):
        
                shutil.make_archive(fname, 'zip', fname)
                with open(f"{fname}.zip", 'rb') as f:
                    res = f.read()
                
                download = st.download_button(label="Download", data=res, key="download", 
                                                file_name=f'{fname}_{int(scale*128)}.zip', 
                                                # disabled=st.session_state.get("disabled", True), 
                                                mime="application/octet-stream")