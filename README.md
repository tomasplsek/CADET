# *Cavity Detection Tool* (CADET)

[CADET](https://tomasplsek.github.io/CADET/) is a machine learning pipeline trained to identify surface brightness depressions (*X-ray cavities*) in noisy *Chandra* images of early-type galaxies and galaxy clusters. The pipeline consists of a convolutional neural network trained to produce pixel-wise cavity predictions and a DBSCAN clustering algorithm that decomposes the predictions into individual cavities. The pipeline is described in detail in [Plšek et al. 2023](https://academic.oup.com/mnras/article/527/2/3315/7339785).

The architecture of the convolutional network consists of 5 convolutional blocks, each resembling an Inception layer, it was implemented using the *Keras* library and its development was inspired by [Fort et al. 2017](https://ui.adsabs.harvard.edu/abs/2017arXiv171200523F/abstract) and [Secká 2019](https://is.muni.cz/th/rnxoz/?lang=en;fakulta=1411). For the clustering, we used is the *Scikit-learn* implementation of the Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

![Architecture](https://github.com/tomasplsek/CADET/raw/main/docs/figures/architecture.png)


## Python package

The CADET pipeline has been released as a standalone Python3 package [`pycadet`](https://pypi.org/project/pycadet/), which can be installed using pip:

```console
$ pip3 install pycadet
```

or from source:

```console
$ pip3 install git+https://github.com/tomasplsek/CADET.git
```

The `pycadet` package requires the following libraries:
```
keras
tensorflow / pytorch / jax
scikit-learn>=1.1
numpy
scipy
astropy
matplotlib
pyds9
```

Since `pycadet v0.3.3`, the package is compatible with Keras3, which supports multiple backends (`tensorflow`, `pytorch` or `jax`). By default, the `tensorflow` is used, but any other backend can be selected by setting the `KERAS_BACKEND` environment variable or editing the `~/keras/keras.json` file. 

The automatic installation of `pycadet` will only install the `keras` package, and the installation of the backend is left to the user. For machines with dedicated NVIDIA graphical cards, the `-gpu` versions of backend libraries (`tensorflow-gpu`, `pytorch-gpu`, `jax-gpu`) can be installed to allow the CADET model to leverage the GPU for faster inference. For Anaconda environments, it is recommended to install the dependencies beforehand as some of the packages can be tricky to install in an existing environment and on some machines (e.g. new Macs).

An exemplary notebook on how to use the `pycadet` package can be found here: 

<a target="_blank" href="https://colab.research.google.com/github/tomasplsek/CADET/blob/main/example/CADET.ipynb">
 <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="margin-bottom:-4px"/>
</a>


## DS9 Plugin

The CADET pipeline can also be used as a [SAOImageDS9](https://ds9.si.edu/) plugin which is installed together with the `pycadet` Python package. The CADET plugin requires that SAOImageDS9 is already installed on the system.

After the installation, the CADET plugin should be available in the *Analysis* menu of DS9. After clicking on the *CADET* option, a new window will appear, where the user can set several options: whether the prediction should be averaged over multiple input images by shifting by +/- 1 pixel (*Shift*); and whether the prediction should be decomposed into individual cavities (*Decompose*). When decomposing into individual cavities, the user can also set a pair of discrimination thresholds, where the first one (*Threshold1*) is used for volume error calibration and the second one (*Threshold2*) for false positive rate calibration (for more info see [Plšek et al. 2023](https://arxiv.org/abs/2304.05457)).

If the CADET plugin does not appear in the *Analysis* menu, it can be added manually by opening *Edit* > *Preferences* > *Analysis* and adding a path to the following file [DS9CADET.ds9.ans](https://github.com/tomasplsek/CADET/raw/main/pycadet/DS9CADET.ds9.ans) (after the installation it should be located in `~/.ds9/`). This plugin is inspired by the [pyds9plugin](https://github.com/vpicouet/pyds9plugin/tree/master) library.

![DS9 CADET plugin](https://github.com/tomasplsek/CADET/raw/main/docs/figures/DS9CADET.gif)

## Online CADET interface

A simplified version of the CADET pipeline is available via a <a href="https://huggingface.co/spaces/Plsek/CADET" target=_blank>web interface</a> hosted on HuggingFace Spaces. The input image should be centred on the galaxy centre and cropped to a square shape. It is also recommended to remove point sources from the image and fill them with the surrounding background level using Poisson statistics ([dmfilth](https://cxc.cfa.harvard.edu/ciao/ahelp/dmfilth.html) within [CIAO](https://cxc.harvard.edu/ciao/)). Furthermore, compared to the `pycadet` package, the web interface performs only a single thresholding of the raw pixel-wise prediction, which is easily adjustable using a slider.

![HuggingFace web interface](https://github.com/tomasplsek/CADET/raw/main/docs/figures/CADET_HF.gif)


## Convolutional part

The convolutional part of the pipeline can be used separately to produce raw pixel-wise predictions. Since the convolutional network was implemented using the functional *Keras* API, the architecture could have been stored together with the trained weights in the HDF5 format ([`CADET.hdf5`](https://github.com/tomasplsek/CADET/raw/main/pycadet/CADET.hdf5)). The trained model can then simply be loaded using the `load_model` *Keras* function (requires *Keras* v2.15 or lower):

```python
from keras.models import load_model

model = load_model("CADET.hdf5")

y_pred = model.predict(X)
```

The raw CADET model only inputs 128x128 images. Furthermore, to maintain the compatibility with *Keras*, the input needs to be reshaped as `X.reshape(1, 128, 128, 1)` for single image or as `X.reshape(-1, 128, 128, 1)` for multiple images.

Alternatively, the CADET model can be imported from HuggingFace's [model hub](https://huggingface.co/Plsek/CADET-v1):

```python
from huggingface_hub import from_pretrained_keras

model = from_pretrained_keras("Plsek/CADET-v1")

y_pred = model.predict(X)
```

## How to cite

If you use the CADET  pipeline in your research, please cite the following paper [Plšek et al. 2023](https://academic.oup.com/mnras/article/527/2/3315/7339785) ([arXiv](https://arxiv.org/abs/2304.05457)):

```
@ARTICLE{2023MNRAS.tmp.3233P,
       author = {{Pl{\v{s}}ek}, T. and {Werner}, N. and {Topinka}, M. and {Simionescu}, A.},
        title = "{CAvity DEtection Tool (CADET): Pipeline for detection of X-ray cavities in hot galactic and cluster atmospheres}",
      journal = {\mnras},
         year = 2023,
        month = nov,
          doi = {10.1093/mnras/stad3371},
}
```

## Todo

The following improvements to the data generation and training process are currently planned:

- [ ] add other features (cold fronts, complex sloshing, point sources, jets)
- [ ] use more complex cavity shapes (e.g. [Guo et al. 2015](https://arxiv.org/abs/1408.5018))
- [ ] train on multiband images simulated using PyXsim/SOXS
- [ ] replace DBSCAN by using instance segmentation 
- [ ] restrict the cavity number and shape using regularization?
- [ ] systematic cavity size uncertainty estimation using MC Dropout
