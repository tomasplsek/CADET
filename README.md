# *Cavity Detection Tool* (CADET)

[CADET](https://tomasplsek.github.io/CADET/) is a machine learning pipeline trained for identification of surface brightness depressions (*X-ray cavities*) on noisy *Chandra* images of early-type galaxies and galaxy clusters. The pipeline consists of a convolutional neural network trained for producing pixel-wise cavity predictions and a DBSCAN clustering algorithm, which decomposes the predictions into individual cavities. The pipeline is further described in [Plšek et al. 2023](https://arxiv.org/abs/2304.05457).

The architecture of the convolutional network consists of 5 convolutional blocks, each resembling an Inception layer, it was implemented using *Keras* library and it's development was inspired by [Fort et al. 2017](https://ui.adsabs.harvard.edu/abs/2017arXiv171200523F/abstract) and [Secká 2019](https://is.muni.cz/th/rnxoz/?lang=en;fakulta=1411). For the clustering, we utilized is the *Scikit-learn* implementation of the Density-Based Spatial Clustering of Applications with Noise (DBSCAN, [Ester et al. 1996](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.121.9220)).

![Architecture](https://github.com/tomasplsek/CADET/raw/main/docs/figures/architecture.png)


## Python package

The CADET pipeline was released as a self-standing Python3 package `pycadet`, which can be installed using pip:

```console
$ pip3 install pycadet
```

To `pycadet` package requieres the following libraries (should be installed automatically with the package):
```
numpy
scipy
astropy
matplotlib
pyds9
scikit-learn>=1.1
tensorflow>=2.8
```

For Conda environments, it is recommended to install the dependencies beforehand as some of the packages can be tricky to install into an existing environment (especially `tensorflow`) and on some machines (especially new Macs). For machines with dedicated NVIDIA GPUs, `tensorflow-gpu` can be installed to allow the CADET model to leverage the GPU for faster inference.

An exemplary notebook on how to use the `pycadet` package can be found here: 

<a target="_blank" href="https://colab.research.google.com/github/tomasplsek/CADET/blob/main/examples/CADET.ipynb">
 <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="margin-bottom:-4px"/>
</a>


<!-- If you want to re-train the network from scratch or generate training images, an additional library is required:\
[`jax`](https://github.com/google/jax) -->

<!-- The CADET pipeline inputs either raw *Chandra* images in units of counts (numbers of captured photons) or exposure-corrected images. When using exposure-corrected images, images should be normalized by the lowest pixel value so all pixels are higher than or equal to 1. For images with many point sources, they should be filled with surrounding background level using Poisson statistics ([dmfilth](https://cxc.cfa.harvard.edu/ciao/ahelp/dmfilth.html) within [CIAO](https://cxc.harvard.edu/ciao/)).

Convolutional part of the CADET pipeline can only input 128x128 images. As a part of the pipeline, input images are therefore being cropped to a size specified by parameter scale (size = scale * 128 pixels) and re-binned to 128x128 images. By default, images are probed on 4 different scales (1,2,3,4). The size of the image inputted into the pipeline therefore needs to at least 512x512 pixels (minimal input size differs if non-default scales are used) and images should be centred at the centre of the galaxy. The re-binning is performed using *Astropy* and *Numpy* libraries and can only handle integer binsizes. For floating point number binning, we recommend using [dmregrid](https://cxc.cfa.harvard.edu/ciao/ahelp/dmregrid.html) and applying CADET model manually (see Convolutional part).

Before being decomposed by the DBSCAN algorithm, pixel-wise predictions produced by the convolutional part of the CADET pipeline need to be further thresholded. In order to simultaneously calibrate the volume error and false positive rate, we introduced two discrimination thresholds (for more info see [Plšek et al. 2023]()) and their default values are 0.4 and 0.6, respectively. Nevertheless, both discrimination thresholds are changeable and can be set to an arbitrary value between 0 and 1.


```python
from pycadet import rebin
```

```python
data, wcs = rebin("NGC4649.fits", scale=2)
```

The `CADET.py` script loads a FITS file specified by the `filename` argument, which is located in the same folder as the main `CADET.py` script. The script creates a folder of the same name as the FITS file, and saves corresponding pixel-wise as well as decomposed cavity predictions into the FITS format while also properly preserving the WCS coordinates. On the output, there is also a PNG file showing decomposed predictions for individual scales.

The volumes of X-ray cavities are calculated under the assumption of rotational symmetry along the direction from the galactic centre towards the centre of the cavity (estimated as *center of mass*). The cavity depth in each point along that direction is then assumed to be equal to its width. Thereby produced 3D cavity models are stored in the `.npy` format and can be used for further caclulation (e.g. cavity energy estimationš) -->

<!-- ![](docs/figures/NGC5813.png) -->

## DS9 Plugin

The CADET pipeline can also be used as a [SAOImageDS9](ds9.si.edu/) plugin which is installed together with the `pycadet` Python package. The CADET plugin requires SAOImageDS9 to be already installed on the system. To avoid conflicts (e.g. the CIAO installation of DS9), it is recommended to install `pycadet` using a system installation of Python3 rather than a Conda environment.

After the installation, the CADET plugin should be available in the *Analysis* menu of DS9. After clicking on the *CADET* option, a new window will appear, where the user can set several options: whether the prediction should be averaged over multiple input images by dithering by +/- 1 pixel (*Dither*); and whether the prediction should be decomposed into individual cavities (*Decompose*). When decomposing into individual cavities, the user can also set a pair of discrimination thresholds, where the first one is used for volume error calibration and the second one for false positive rate calibration (for more info see [Plšek et al. 2023](https://arxiv.org/abs/2304.05457)).

If the CADET plugin does not appear in the *Analysis* menu, it can be added manually by opening *Edit* > *Preferences* > *Analysis* and adding a path to the following file [DS9CADET.ds9.ans](https://github.com/tomasplsek/CADET/raw/main/pycadet/DS9CADET.ds9.ans) (after the installation it should be located in `~/.ds9/`). The plugin is inspired by the [pyds9plugin](https://github.com/vpicouet/pyds9plugin/tree/master) library.

![HuggingFace web interface](https://github.com/tomasplsek/CADET/raw/main/docs/figures/DS9CADET.gif)



## Online CADET interface

The CADET pipeline can also be accessed through a simple [web interface](https://huggingface.co/spaces/Plsek/CADET) hosted on HuggingFace Spaces.

![HuggingFace web interface](https://github.com/tomasplsek/CADET/raw/main/docs/figures/CADET_Huggingface.png)


## Convolutional part

<!-- [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tomasplsek/CADET/blob/main/CADET_example_colab.ipynb) -->

The convolutional part of the pipeline can be used separately to produce raw pixel-wise predictions. Since the convolutional network was implemented using the functional *Keras* API, the architecture together with trained weights could have been stored in the HDF5 format ([`CADET.hdf5`](https://github.com/tomasplsek/CADET/raw/main/CADET.hdf5)). The trained model can be therefore simply loaded using the `load_model` *TensorFlow* function:

```python
from tensorflow.keras.models import load_model

model = load_model("CADET.hdf5")

y_pred = model.predict(X)
```

The input image 128x128 images. To maintain the compatibility with *Keras*, the input needs to be reshaped as `X.reshape(1, 128, 128, 1)` for single image or as `X.reshape(-1, 128, 128, 1)` for multiple images.

Alternatively, the CNN model can be imported from HuggingFace's [model hub](https://huggingface.co/Plsek/CADET-v1):

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
