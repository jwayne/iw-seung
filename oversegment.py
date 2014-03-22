from __future__ import division
from jpyutils import importer, io, jargparse
import matplotlib.pyplot as plt
import numpy as np
import os

import config
import formats



def plot_oversegmented(labels_2d):
    plot_arr(labels_2d, plt.cm.spectral)
    plt.show()

def plot_arr(arr, cmap):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(arr, cmap=cmap, interpolation='nearest')



def main(oversegmenter, to_plot=False, lim=0):

    module = importer.get_module('oversegmenters', oversegmenter)

    if hasattr(module, 'oversegment_bm'):
        bm_3d = formats.read_bm(config.fn_bm)
        if lim:
            bm_3d = bm_3d[:lim]
        labels_3d, n_labels = module.oversegment_bm(bm_3d)
    elif hasattr(module, 'oversegment_aff'):
        aff_3d = formats.read_aff(config.fn_aff)
        if lim:
            aff_3d = aff_3d[:lim]
        labels_3d, n_labels = module.oversegment_aff(aff_3d)
    else:
        raise ImportError("Bad module: '%s'" % module.__name__)
    # Standardize..
    labels_3d = np.uint16( labels_3d ) 

    if to_plot:
        for labels_2d in labels_3d:
            plot_oversegmented(labels_2d)

    # Write labels to disk
    fn = io.get_filename(config.dn_data, "oversegment-%s" % oversegmenter, "tif")
    formats.save_labels(fn, labels_3d)



if __name__ == "__main__":
    parser = jargparse.ArgumentParser()
    parser.add_argument('oversegmenter')
    parser.add_argument('--lim', type=int, default=0,
        help="max slices to oversegment")
    parser.add_argument('--plot', action='store_true',
        help="plot oversegmented labels for each slice")
    args = parser.parse_args()

    main(args.oversegmenter, args.plot, args.lim)
