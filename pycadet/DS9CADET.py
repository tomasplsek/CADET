import argparse, sys
from pycadet import make_prediction, make_3D_cavity

import os
os.environ["KERAS_BACKEND"] = "torch"


######################## THIS PART OF CODE IS FROM PYDS9PLUGIN ########################
####################### https://github.com/vpicouet/pyds9plugin #######################

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

########################################################################################


def CADET(argv=[]):
    import numpy as np
    from scipy.ndimage import zoom
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.nddata import Cutout2D
    from sklearn.cluster import DBSCAN

    # Parse arguments
    parser = CreateParser(get_name_doc(),path=True)
    parser.add_argument('-shift', '--shift', help='Shift the input region by +/- 1 pixel (increases execution time ~9 times).', default='0', metavar='')
    # parser.add_argument('-b', '--bootstrap', help='Boostrap the input image (increases execution time ~n times).', default='0', metavar='')
    # parser.add_argument('-n', '--bootstrap_n', help='Number of bootstrap iterations per single rotation-shifting configuration.', default=1, type=int, metavar='')
    parser.add_argument('-dec', '--decompose', help='Decompose raw cavity prediction into individual cavities.', default='0', metavar='')
    parser.add_argument('-th1', '--threshold1', help='Volume calibrating threshold.', default=0.4, type=float, metavar='')
    parser.add_argument('-th2', '--threshold2', help='TP/FP calibrating threshold.', default=0.6, type=float, metavar='')
    parser.add_argument('-sf', '--sf', help='Save files.', default='0', metavar='')
    args = parser.parse_args_modif(argv,required=True)

    decompose_cavities = bool(int(args.decompose))
    shift = bool(int(args.shift))
    # bootstrap = bool(int(args.bootstrap))
    # N_bootstrap = args.bootstrap_n
    threshold1 = args.threshold1
    threshold2 = args.threshold2
    save = bool(int(args.sf))
    # method = 4

    # Minimal image size
    min_size = 130 if shift else 128

    # Load current DS9 session
    d = DS9n(args.xpapoint)

    # Load current image
    hdu = d.get_pyfits()[0]
    image = hdu.data
    wcs = WCS(hdu.header)

    # Get the filename
    fname = d.get('file')
    fname = fname.split("/")[-1].split(".")[0]

    # Remove negative values (XMM-Newton data)
    image = np.where(image > 0, image, 0)

    # Load current region file
    d.set("regions format ds9")
    d.set("regions system image")
    region = d.get('regions')
    lines = region.split("\n")
    box = None
    for line in lines:
        if "box" in line:
            box = line
            break

    if box is None:
        print("No box region found.", end=" ")

        if abs(image.shape[0] - image.shape[1]) > 1:
            print("Image does not seem to be cropped.")
            print("\nPlease select a box region or crop the image into a square.")
            return
        else:
            print("Using the full image.")

            w, h = image.shape[0], image.shape[1]
            x, y = w // 2, h // 2
            scale = min([w,h]) // min_size
            size = min_size * scale

            print(f"Size of selected region:  {w}x{h} pixels")

            if (w < min_size) or (h < min_size):
                print(f"Image is smaller than the minimal size of {min_size}x{min_size} pixels.")
                print(f"Please select a bigger image.")
                return

            print(f"Cropping input image to:  {size}x{size} pixels")

    else:
        print("Using selected box region.")

        dims = box.split("(")[1][:-1].split(",")
        # ra = float(dims[0])
        # dec = float(dims[1])
        # w = int(round(float(dims[2][:-1]) / pixel_size))
        # h = int(round(float(dims[3][:-1]) / pixel_size))

        x = float(dims[0])
        y = float(dims[1])
        w = int(round(float(dims[2])))
        h = int(round(float(dims[3])))

        print(f"Size of selected region:  {w}x{h} pixels")

        if (w < min_size) or (h < min_size):
            print(f"Region is smaller than the minimal size of {min_size}x{min_size} pixels.")
            print(f"Please select a bigger region.")
            return

        # x, y = wcs.all_world2pix(ra, dec, 0)
        x, y = int(round(float(x))), int(round(float(y)))
        scale = min([h,w]) // min_size
        size = min_size * scale

        print(f"Cropping input image to:  {size}x{size} pixels")

    # Get angular size of 1 pixel in arcsec
    angular_scale = wcs.pixel_scale_matrix[1,1] * 3600

    # Crop the image & wcs
    cutout = Cutout2D(image, (x, y), (size, size), wcs=wcs)
    image, wcs = cutout.data, cutout.wcs
    hdu = fits.PrimaryHDU(image, header=wcs.to_header())

    print("Rebinning input image to: 128x128 pixels")
    image = image.reshape(min_size, scale, min_size, scale).sum(-1).sum(1)

    print("\nApplying CADET model...", end=" ")
    y_pred = make_prediction(image, shift=shift) #, bootstrap=bootstrap, N_bootstrap=N_bootstrap)
    print("Success.")

    y_pred = zoom(y_pred, (scale, scale))

    hdu.data = y_pred
    hdu.writeto(f"/tmp/CADET_raw.fits", overwrite=True)
    if save: hdu.writeto(f"{fname}_CADET_raw.fits", overwrite=True)

    # Load the image into DS9
    d.set("frame new ; tile yes ; file " + f"/tmp/CADET_raw.fits")
    d.set("frame lock wcs")
    d.set('scale linear')
    d.set('cmap viridis')
    d.set('contour yes')
    d.set('contour levels "0.4 0.6 0.9"')
    d.set('contour color white')
    d.set('contour width 2')
    d.set('contour save /tmp/CADET_contours.ctr wcs fk5')
    d.set('contour close')
    d.set('contour clear')

    # Decompose into individual cavities
    if decompose_cavities:
        print(f"\nDecomposing into individual cavities (th1={threshold1}, th2={threshold2}):")
        
        # THRESHOLDING
        y_pred = np.where(y_pred > threshold1, y_pred, 0)

        # Decompose
        X, Y = y_pred.nonzero()
        data = np.array([X,Y]).reshape(2, -1)

        # DBSCAN CLUSTERING ALGORITHM
        try: clusters = DBSCAN(eps=1.5, min_samples=3).fit(data.T).labels_
        except: clusters = []

        N = len(set(clusters))

        n = 0
        cavs = np.zeros((128*scale,128*scale))
        for i in range(N):
            img = np.zeros((128*scale,128*scale))
            b = clusters == i
            xi, yi = X[b], Y[b]
            img[xi, yi] = y_pred[xi, yi]

            # THRESHOLDING #2
            if not (img > threshold2).any(): continue

            cav = np.where(img > 0, 1, 0)
            cube = make_3D_cavity(zoom(cav, (1/scale, 1/scale)))

            n += 1
            print(f"\nCavity {n}:\narea  =  {np.sum(cav):.0f} pixels² ({np.sum(cav)*(angular_scale)**2:.0f} arcsec²)")
            print(f"volume = {np.sum(cube)*scale**3:.0f} pixels³ ({np.sum(cube)*(scale*angular_scale)**3:.0f} arcsec³)")
            cavs += cav * n

        hdu.data = cavs
        hdu.writeto(f"/tmp/CADET_decomposed.fits", overwrite=True)
        if save: hdu.writeto(f"{fname}_CADET_decomposed.fits", overwrite=True)

        # Load the image into DS9
        d.set("frame new ; tile column ; file " + f"/tmp/CADET_decomposed.fits")
        d.set('cmap i8')
        d.set("frame lock wcs")

    d.set("frame next")
    d.set('contour load /tmp/CADET_contours.ctr')

    return


def main():
    """Main function where the arguments are defined and the other functions called
    """

    DictFunction = {"CADET" : CADET}

    function = sys.argv[1]

    if sys.stdin is None:
        try:
            DictFunction[function]()
        except Exception as e:
            pass
    else:
        DictFunction[function]()

    return

if __name__ == "__main__":
    a = main()
