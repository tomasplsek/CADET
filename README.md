# *Cavity Detection Tool* (CADET)

[CADET](https://tomasplsek.github.io/CADET/) is a machine learning pipeline trained to identify of surface brightness depressions (*X-ray cavities*) in noisy *Chandra* images of early-type galaxies and galaxy clusters. The pipeline consists of a convolutional neural network trained to produce pixel-wise cavity predictions and a DBSCAN clustering algorithm that decomposes the predictions into individual cavities. The pipeline is described in detail in [Plšek et al. 2023](https://arxiv.org/abs/2304.05457).

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

The `pycadet` package requires the following libraries (which should be installed automatically with the package):
```
numpy
scipy
astropy
matplotlib
pyds9
scikit-learn>=1.1
tensorflow>=2.8
```

For Conda environments, it is recommended to install the dependencies beforehand as some of the packages can be tricky to install in an existing environment (especially `tensorflow`) and on some machines (especially new Macs). For machines with dedicated NVIDIA GPUs, `tensorflow-gpu` can be installed to allow the CADET model to leverage the GPU for faster inference.

An exemplary notebook on how to use the `pycadet` package can be found here: 

<a target="_blank" href="https://colab.research.google.com/github/tomasplsek/CADET/blob/main/example/CADET.ipynb">
 <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="margin-bottom:-4px"/>
</a>


## DS9 Plugin

The CADET pipeline can also be used as a [SAOImageDS9](ds9.si.edu/) plugin which is installed together with the `pycadet` Python package. The CADET plugin requires that SAOImageDS9 is already installed on the system. To avoid conflicts (e.g. the CIAO installation of DS9), it is recommended to install `pycadet` using a system installation of Python3 rather than a Conda environment.

After the installation, the CADET plugin should be available in the *Analysis* menu of DS9. After clicking on the *CADET* option, a new window will appear, where the user can set several options: whether the prediction should be averaged over multiple input images by shifting by +/- 1 pixel (*Shift*); and whether the prediction should be decomposed into individual cavities (*Decompose*). When decomposing into individual cavities, the user can also set a pair of discrimination thresholds, where the first one (*Threshold1*) is used for volume error calibration and the second one (*Threshold2*) for false positive rate calibration (for more info see [Plšek et al. 2023](https://arxiv.org/abs/2304.05457)).

If the CADET plugin does not appear in the *Analysis* menu, it can be added manually by opening *Edit* > *Preferences* > *Analysis* and adding a path to the following file [DS9CADET.ds9.ans](https://github.com/tomasplsek/CADET/raw/main/pycadet/DS9CADET.ds9.ans) (after the installation it should be located in `~/.ds9/`). The plugin is inspired by the [pyds9plugin](https://github.com/vpicouet/pyds9plugin/tree/master) library.

![DS9 CADET plugin](https://github.com/tomasplsek/CADET/raw/main/docs/figures/DS9CADET.gif)



## Online CADET interface

A simplified version of the CADET pipeline is available via a <a href="https://huggingface.co/spaces/Plsek/CADET" target=_blank>web interface</a> hosted on HuggingFace Spaces. The input image should be centred on the galaxy centre and cropped to a square shape. It is also recommended to remove point sources from the image and fill them with the surrounding background level using Poisson statistics ([dmfilth](https://cxc.cfa.harvard.edu/ciao/ahelp/dmfilth.html) within [CIAO](https://cxc.harvard.edu/ciao/)). Furthermore, compared to the `pycadet` package, the web interface performs only a single thresholding of the raw pixel-wise prediction, which is easily adjustable using a slider.

![HuggingFace web interface](https://github.com/tomasplsek/CADET/raw/main/docs/figures/CADET_HF.gif)


## Convolutional part

<!-- [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tomasplsek/CADET/blob/main/CADET_example_colab.ipynb) -->

The convolutional part of the pipeline can be used separately to produce raw pixel-wise predictions. Since the convolutional network was implemented using the functional *Keras* API, the architecture could have been stored together with the trained weights in the HDF5 format ([`CADET.hdf5`](https://github.com/tomasplsek/CADET/raw/main/CADET.hdf5)). The trained model can then simply be loaded using the `load_model` *TensorFlow* function:

```python
from tensorflow.keras.models import load_model

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

<!-- Thus produced pixel-wise prediction needs to be further thresholded and decomposed into individual cavities using a DBSCAN clustering algorithm:

```python
import numpy as np
from sklearn.cluster import DBSCAN

y_pred = np.where(y_pred > threshold, 1, 0)

x, y = y_pred.nonzero()
data = np.array([x,y]).reshape(2, -1)

clusters = DBSCAN(eps=1.5, min_samples=3).fit(data.T).labels_
``` -->

## How to cite

<!-- The CADET pipeline is thoroughly described in [Plšek et al. 2023](https://arxiv.org/abs/2304.05457) and was originally developed as a part of my [diploma thesis](https://is.muni.cz/th/x68od/?lang=en).  -->
If you use the CADET  pipeline in your research, please cite the following paper [Plšek et al. 2023](https://arxiv.org/abs/2304.05457):

```
@misc{plšek2023cavity,
      title={CAvity DEtection Tool (CADET): Pipeline for automatic detection of X-ray cavities in hot galactic and cluster atmospheres}, 
      author={Tomáš Plšek and Norbert Werner and Martin Topinka and Aurora Simionescu},
      year={2023},
      eprint={2304.05457},
      archivePrefix={arXiv},
      primaryClass={astro-ph.HE}
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
