import argparse
import os, sys

from pycadet import rebin, make_prediction, decompose

# try:
#     import IPython.core.ultratb
# except ImportError:
#     pass
# else:
#     sys.excepthook = IPython.core.ultratb.ColorTB()



def get_name_doc():
    import inspect
    outerframe = inspect.currentframe().f_back
    name = outerframe.f_code.co_name
    doc = outerframe.f_back.f_globals[name].__doc__
    return name, doc



class MyParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help()
        sys.stdout.write('error: %s\n' % message)
        sys.exit(2)
    def parse_args_modif(self, argv, required=True):
        if len(argv)==0:
            args = self.parse_args()
        else:
            args = self.parse_args(['test']+argv.split())
        if hasattr(args, "path") is False:
            args.path = None
        if required & (args.xpapoint is None) & ((args.path is None)|(args.path =='')):
            self.error("at least one of --xpapoint and --path required")
        return args


def CreateParser(namedoc,path=False):
    name, doc = namedoc
    formatter = lambda prog: argparse.HelpFormatter(prog,max_help_position=32,width=136) #argparse.ArgumentDefaultsHelpFormatter
    parser = MyParser(description='%s: %s'%(name, doc),usage="DS9CADET %s [-h] [-x xpapoint] [--optional OPTIONAL]"%(name),formatter_class=formatter)
    parser.add_argument('function', help="Function to perform [here: %s]"%(name))#,required=True)
    parser.add_argument('-x', '--xpapoint', help='XPA access point for DS9 communication. If none is provided, it will take the last DS9 window if one, else it will run the function without DS9.', metavar='')
    if path:
        parser.add_argument('-p', '--path', help='Path of the image(s) to process, regexp accepted', metavar='',default='')
    return parser



class FakeDS9(object):
    def __init__(self, **kwargs):
        """For sharing a porfoilio
        """
        self.total = []

    def get(self, value=''):
        return True
    def set(self, value=''):
        return True


def DS9n(xpapoint=None, stop=False):
    """Open a DS9 communication with DS9 software, if no session opens a new one
    else link to the last created session. Possibility to give the ssession
    you want to link"""
    from pyds9 import DS9, ds9_targets

    targets = ds9_targets()
    if targets:
        xpapoints = [target.split(" ")[-1] for target in targets]
    else:
        xpapoints = []
    if ((xpapoint == 'None') | (xpapoint is None)) & (len(xpapoints) == 0):
        return FakeDS9()
    elif len(xpapoints) != 0:
        # verboseprint("%i targets found" % (len(xpapoints)))
        if xpapoint in xpapoints:
            pass
            # verboseprint("xpapoint %s in targets" % (xpapoint))
        else:
            if stop:
                sys.exit()
            else:
                # verboseprint("xpapoint %s NOT in targets" % (xpapoint))
                xpapoint = xpapoints[0]

    try:
        # verboseprint("DS9(%s)" % (xpapoint))
        d = DS9(xpapoint)
        return d
    except (FileNotFoundError, ValueError) as e:
        d = DS9()



def CADET(argv=[]):
    import numpy as np
    from scipy.ndimage import zoom
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.nddata import Cutout2D, CCDData
    from sklearn.cluster import DBSCAN
    from tensorflow.keras.models import load_model

    path = os.path.dirname(__file__)
    model = load_model(f'{path}/CADET.hdf5')

    # Parse arguments
    parser = CreateParser(get_name_doc(),path=True)
    parser.add_argument('-shift', '--shift', help='Shift the input region by +/- 1 pixel (increases execution time ~9 times).', default='0', metavar='')
    # parser.add_argument('-b', '--bootstrap', help='Boostrap the input image (increases execution time ~n times).', default='0', metavar='')
    # parser.add_argument('-n', '--bootstrap_n', help='Number of bootstrap iterations per single rotation-shifting configuration.', default=1, type=int, metavar='')
    parser.add_argument('-dec', '--decompose', help='Decompose raw cavity prediction into individual cavities.', default='0', metavar='')
    parser.add_argument('-th1', '--threshold1', help='Volume calibrating threshold.', default=0.5, type=float, metavar='')
    parser.add_argument('-th2', '--threshold2', help='TP/FP calibrating threshold.', default=0.9, type=float, metavar='')
    args = parser.parse_args_modif(argv,required=True)

    decompose = bool(int(args.decompose))
    shift = bool(int(args.shift))
    # bootstrap = bool(int(args.bootstrap))
    # N_bootstrap = args.bootstrap_n
    threshold1 = args.threshold1
    threshold2 = args.threshold2
    # method = 4
    pixel_size = 0.492

    min_size = 130 if shift else 128

    # Load current DS9 session
    d = DS9n(args.xpapoint)

    # Load current image
    hdu = d.get_pyfits()[0]
    image = hdu.data
    wcs = WCS(hdu.header)

    # Load current region file
    d.set("regions format ds9")
    d.set("regions system wcs")
    region = d.get('regions')
    lines = region.split("\n")
    box = None
    for line in lines:
        if "box" in line:
            box = line
            break

    if box is None:
        print("No box region found.", end=" ")

        if (image.shape[0] > image.shape[1]+1) | (image.shape[0] < image.shape[1]-1):
            print("Image does not seem to be cropped. Exiting.")
            return
        else:
            print("Using the full image.")

        w, h = image.shape[0], image.shape[1]
        x, y = w // 2, h // 2
        scale = min([w,h]) // min_size
        size = min_size * scale

        print(f"The shape of the region is {w}x{h} pixels.")
        print(f"Cropping the input region to {size}x{size} pixels.")


    else:
        print("Using selected box region.")

        dims = box.split("(")[1][:-1].split(",")
        ra = float(dims[0])
        dec = float(dims[1])
        w = int(round(float(dims[2][:-1]) / pixel_size))
        h = int(round(float(dims[3][:-1]) / pixel_size))

        if (w < min_size) or (h < min_size):
            print(f"Region is smaller than the minimal size of {min_size}x{min_size} pixels.")
            print(f"Please select a bigger region.")
            return

        x, y = wcs.all_world2pix(ra, dec, 0)
        x, y = int(round(float(x))), int(round(float(y)))
        scale = min([h,w]) // min_size
        size = min_size * scale

        print(f"The shape of the region is {w}x{h} pixels.")
        print(f"Cropping the input region to {size}x{size} pixels.")


    cutout = Cutout2D(image, (x, y), (size, size), wcs=wcs)
    image, wcs = cutout.data, cutout.wcs
    hdu = fits.PrimaryHDU(image, header=wcs.to_header())

    print("Rebinning the input region to 128x128 pixels.")
    image = image.reshape(min_size, scale, min_size, scale).sum(-1).sum(1)

    # rotations = [0,1,2,3]
    # if shift:
    #     DX = np.array([0,0,0,1,1,1,-1,-1,-1])
    #     DY = np.array([0,1,-1,0,1,-1,0,1,-1])
    #     # DX = np.array([0,0,0,0,0,1,1,1,1,1,-1,-1,-1,-1,-1,2,2,2,-2,-2,-2])
    #     # DY = np.array([0,1,-1,2,-2,0,1,-1,2,-2,0,1,-1,2,-2,0,1,-1,0,1,-1])
    #     W = 1 / (1 + (DX**2 + DY**2)**0.5)
    # else:
    #     DX = np.array([0])
    #     DY = np.array([0])
    #     W = np.array([1])

    # N_total = len(rotations) * len(DX) * N_bootstrap


    # def bootstrap(data, variance=False):
    #     if variance: data = np.random.poisson(data+1)

    #     x0, y0 = data.nonzero()
    #     probs = data[x0,y0]
    #     indices = np.arange(len(x0))

    #     sampled_indices = np.random.choice(a=indices, size=int(np.sum(data)), p=probs/np.sum(probs), replace=True)

    #     i, v = np.unique(sampled_indices, return_counts=True)

    #     sampled_data = np.zeros(data.shape)
    #     sampled_data[x0[i],y0[i]] = v

    #     return sampled_data

    # X = []
    # for dx, dy, w in zip(DX, DY, W):
    #     x0 = image.shape[0] / 2
    #     cutout = Cutout2D(image, (x0+dx, x0+dy), (128, 128), wcs=wcs)
    #     image_cut = cutout.data

    #     # ROTATIONAL AVERAGING
    #     for j in rotations:
    #         rotated0 = np.rot90(image_cut, j)
    #         for n in range(N_bootstrap):
    #             if N_bootstrap > 1:
    #                 if method == 1: rotated = np.random.poisson(rotated0)
    #                 elif method == 2: rotated = rotated0 + np.random.poisson(np.ones((128,128))*fac)
    #                 elif method == 3: rotated = np.random.poisson(rotated0 + fac)
    #                 elif method == 4: rotated = bootstrap(rotated0)
    #                 elif method == 5: rotated = bootstrap(rotated0, variance=True)
    #                 else: rotated = rotated0
    #             else: rotated = rotated0

    #             X.append(np.log10(rotated+1))

    # # APPLY CADET MODEL (runs faster if applied on all images at once, hence the two for loops)
    # preds = model.predict(np.array(X).reshape(N_total, 128, 128, 1), verbose=0).reshape(N_total, 128 ,128)

    # # SHIFT IMAGES BACK TO ORIGINAL POSITION
    # y_pred = np.zeros((N_total, 128, 128))
    # x = 0
    # for dx, dy, w in zip(DX, DY, W):
    #     for j in rotations:
    #         for n in range(N_bootstrap):
    #             pred = np.rot90(preds[x], -j)
    #             prediction = np.zeros((128,128))
    #             prediction[max(0,dy):128+min(0,dy), max(0,dx):128+min(0,dx)] = pred[max(0,-dy):128+min(0,-dy), max(0,-dx):128+min(0,-dx)]
    #             y_pred[x,:,:] += prediction * w / sum(W)
    #             x += 1

    # # print(y_pred.shape)
    # # y_std = np.std(y_pred, axis=0) / N_bootstrap / len(rotations)
    # y_pred = np.sum(y_pred, axis=0) / N_bootstrap / len(rotations)

    print("Applying CADET pipeline.")
    y_pred = make_prediction(image, shift=shift) #, bootstrap=bootstrap, N_bootstrap=N_bootstrap)

    y_pred = zoom(y_pred, (scale, scale))

    hdu.data = y_pred
    hdu.writeto("/tmp/CADET_raw.fits", overwrite=True)

    # Load the image into DS9
    d.set("frame new ; tile yes ; file " + "/tmp/CADET_raw.fits")
    d.set("frame lock wcs")
    d.set('scale linear')
    d.set('cmap viridis')
    d.set('contour yes')
    d.set('contour levels "0.5 0.8 0.9"')
    d.set('contour color white')
    d.set('contour width 2')
    d.set('contour save /tmp/CADET_contours.ctr wcs fk5')
    d.set('contour close')
    d.set('contour clear')

    if decompose:
        # THRESHOLDING
        y_pred = np.where(y_pred > threshold1, y_pred, 0)

        # Decompose
        X, Y = y_pred.nonzero()
        data = np.array([X,Y]).reshape(2, -1)

        # DBSCAN CLUSTERING ALGORITHM
        try: clusters = DBSCAN(eps=1.5, min_samples=3).fit(data.T).labels_
        except: clusters = []

        N = len(set(clusters))
        n = 1
        cavs = np.zeros((128*scale,128*scale))
        for i in range(N):
            img = np.zeros((128*scale,128*scale))
            b = clusters == i
            xi, yi = X[b], Y[b]
            img[xi, yi] = y_pred[xi, yi]

            # THRESHOLDING #2
            if not (img > threshold2).any(): continue
            cavs += np.where(img > 0, 1, 0) * n
            n += 1

        hdu.data = cavs
        hdu.writeto("/tmp/CADET_decomposed.fits", overwrite=True)

        # Load the image into DS9
        d.set("frame new ; tile column ; file " + "/tmp/CADET_decomposed.fits")
        d.set('cmap i8')
        d.set("frame lock wcs")

    d.set("frame next")
    d.set('contour load /tmp/CADET_contours.ctr')

    return


def main():
    """Main function where the arguments are defined and the other functions called
    """

    DictFunction = {
        "CADET" : CADET,
    }

    function = sys.argv[1]

    if sys.stdin is None:
        try:
            DictFunction[function]()#(xpapoint=xpapoint)
        except Exception as e:  # Exception #ValueError #SyntaxError
            pass
    else:
        DictFunction[function]()#(xpapoint=xpapoint)

    return

if __name__ == "__main__":
    a = main()
