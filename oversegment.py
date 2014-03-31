#!/usr/bin/env python
from __future__ import division
from jpyutils import importer, io, jargparse
import matplotlib.pyplot as plt
import numpy as np
import os

import config
from structs import formats



def plot_oversegmented(labels_2d):
    plot_arr(labels_2d, plt.cm.spectral)
    plt.show()

def plot_arr(arr, cmap):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(arr, cmap=cmap, interpolation='nearest')



def oversegment(oversegmenter, lim=0, infile=None):
    module = importer.get_module('oversegmenters', oversegmenter)

    if hasattr(module, 'oversegment_bm'):
        if infile:
            fn = infile
        else:
            fn = config.fn_bm
        bm_3d = formats.read_bm(fn)
        if lim:
            bm_3d = bm_3d[:lim]
        labels_3d, n_labels = module.oversegment_bm(bm_3d)

    elif hasattr(module, 'oversegment_aff'):
        if infile:
            fn = infile
        else:
            fn = config.fn_aff
        aff_3d = formats.read_aff(fn)
        if lim:
            aff_3d = aff_3d[:lim]
            # Need edges going out-of-bounds to be 0, to prevent the exploring
            # of out-of-bounds vertices
        formats.clean_aff(aff_3d)
        labels_3d, n_labels = module.oversegment_aff(aff_3d)

    else:
        raise ImportError("Bad module: '%s'" % module.__name__)

    logging.info("Found %d labels" % n_labels)

    # Write labels to disk
    fn = io.get_filename(config.dn_data, "oversegment-%s" % oversegmenter, "tif")
    formats.save_labels(fn, labels_3d)


def main():
    parser = jargparse.ArgumentParser()
    parser.add_argument('oversegmenter')
    parser.add_argument('--infile',
        help="image file to segment, if not supplied then file in config.py will be used")
    parser.add_argument('--lim', type=int, default=0,
        help="max slices to oversegment")
    args = parser.parse_args()

    oversegment(args.oversegmenter, args.lim, args.infile)


if __name__ == "__main__":
    args = main()
