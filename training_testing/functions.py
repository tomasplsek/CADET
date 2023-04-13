import glob, io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import center_of_mass, rotate

import tensorflow as tf
from sklearn.cluster import DBSCAN

fsize = 15
plt.rc('font', size=fsize)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{newtxtext}\usepackage{newtxmath}')


def decompose(pred, threshold2=0.7, amin=10):
    X, Y = pred.nonzero()
    data = np.array([X,Y]).reshape(2, -1)

    try: 
        clusters = DBSCAN(eps=1, min_samples=3).fit(data.T).labels_
    except:
        clusters = []

    N = len(set(clusters))
    cavs, xn, yn, clustersn = [], [], [], []
    if N < 15:
        for j in range(N):
            img = np.zeros((128,128))
            b = clusters == j
            xi, yi = X[b], Y[b]
            img[xi,yi] = pred[xi, yi]

            # THRESHOLDING #2
            if not (img >= threshold2).any(): continue

            # MINIMAL AREA
            if np.sum(img) <= amin: continue

            xn = np.concatenate((xn, xi))
            yn = np.concatenate((yn, yi))
            clustersn = np.concatenate((clustersn, clusters[b]))
            cavs.append(img)
    else:
        img = np.zeros((128,128))
        
    return cavs, xn, yn, clustersn


def decompose_two(pred, threshold2=0.7, amin = 10):
    X, Y = pred.nonzero()
    data = np.array([X,Y]).reshape(2, -1)

    try:
        clusters = DBSCAN(eps=1, min_samples=3).fit(data.T).labels_
    except:
        clusters = []

    N = len(set(clusters))
    cavs = []
    clus, occ = np.unique(clusters, return_counts=True)
    biggest = np.argsort(occ)[-2:]

    for j in range(N):
        if j in biggest:
            img = np.zeros((128,128))
            b = clusters == j
            xi, yi = X[b], Y[b]
            img[xi,yi] = pred[xi, yi]

            # THRESHOLDING #2
            if not (img >= threshold2).any(): continue

            # MINIMAL AREA
            if np.sum(img) <= amin: continue

            cavs.append(img)
        
    return cavs


def make_cube(image, imin=0.1):
    cen = center_of_mass(image)
    phi = np.arctan2(cen[0]-63.5, cen[1]-63.5)
    image = rotate(image, phi*180/np.pi, reshape=False, prefilter=False)
    image = np.where(image > imin, 1, 0)

    cube = np.zeros((128,128,128))
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

    for m, w, i in zip(means, widths, indices):
        x, y = np.indices((128, 128))
        r = np.sqrt((x-abs(m))**2 + (y-63.5)**2)
        sliced = np.where(r <= w, 1, 0)
        cube[:,:,i] += sliced

    return cube


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches="tight")
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)


def exponential(p1=0.15, p2= 0.6, size=100):
    X = []
    while len(X) < size:
        x = np.random.exponential(p1)
        if x < p2: X.append(x)
    return X


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def histogram(array, bins, norm=False):
    h = np.histogram(array, bins = bins)
    x = (h[1][1:]+h[1][:-1])/2
    if norm: return x, x[1]-x[0], h[0] / len(bins)
    else: return x, x[1]-x[0], h[0]


def make_hist(X, column, bins=20, log=False):
    plt.rc('font', size=size)

    kw = {"fc" : (31/255, 119/255, 180/255, 0.5),
          "edgecolor" : "C0",
          "color" : None,
          "lw" : 2}
    
    if not log:
        bins = np.linspace(min(X), max(X), bins)
    elif log:
        bins = np.logspace(np.log10(min(X)), np.log10(max(X)), bins)
        
    fig, ax = plt.subplots()
    ax.hist(X, bins, **kw)
    
    if log: ax.set_xscale("log")
    ax.set_xlabel(column);
    
    x_formatter = FuncFormatter(lambda y, _: "{:.0f}".format(y))
    y_formatter = x_formatter
    if ax.get_xscale() == "log":
        x_formatter = FuncFormatter(lambda y, _: "10$^{{\\text{{{:.2g}}}}}$".format(np.log10(y)))
    ax.xaxis.set_major_formatter(x_formatter)
    ax.yaxis.set_major_formatter(y_formatter)
    
    return fig, ax


def rround(x,n):
    y = round(x,n)
    if (y % 1) == 0: return int(y)
    else: return y


def texg(x, xe, n):
    if x == 0: return "0"
    if xe[0] == x: xe = np.array([0.9, 1.1]) * x
    r = -(int(np.log10(x-xe[0])) - n)
    if round(x-xe[0],r) == round(xe[1]-x,r):
        return "${0}\pm{1}$".format(rround(x,r), rround(x-xe[0],r))
    return "${0}_{{-{1}}}^{{+{2}}}$".format(rround(x,r), rround(x-xe[0],r), rround(xe[1]-x,r))


def texf(x, xe, n=1):
    if n != 0:
        if round(x-xe[0],n) == round(xe[1]-x,n):
            return "${0}\pm{1}$".format(round(x,n), round(x-xe[0],n))
        return "${0}_{{-{1}}}^{{+{2}}}$".format(round(x,n), round(x-xe[0],n), round(xe[1]-x,n))
    elif n == 0:
        if round(x-xe[0],n) == round(xe[1]-x,n):
            return "${0:.0f}\pm{1:.0f}$".format(round(x,n), round(x-xe[0],n))
        return "${0:.0f}_{{-{1:.0f}}}^{{+{2:.0f}}}$".format(round(x,n), round(x-xe[0],n), round(xe[1]-x,n))

