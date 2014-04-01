"""
Run a selected oversegmenter algorithm defined as a module in
`oversegmenters/`, on a 3D boundary map or a 3D affinity graph.

For a module defined in `oversegmenters/alg.py`, if
`alg.oversegment_bm` exists then a boundary map is expected as
input.  On the other hand, if `alg.oversegment_aff` exists then
an affinity graph is expected as input.
"""
#!/usr/bin/env python
from __future__ import division
from jpyutils import importer, io, jargparse
import logging
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



def oversegment(oversegmenter, lim=0, infile=None, outfile=None):
    module = importer.get_module('oversegmenters', oversegmenter)
    use_bm = hasattr(module, 'oversegment_bm')
    use_aff = hasattr(module, 'oversegment_aff')
    if use_bm and use_aff:
        raise ImportError("Bad module: '%s'" % module.__name__)

    if use_bm:
        if not infile:
            infile = config.fn_bm
        bm_3d = formats.read_bm(infile)
        if lim:
            bm_3d = bm_3d[:lim]
        labels_3d, n_labels = module.oversegment_bm(bm_3d)

    elif use_aff:
        if not infile:
            infile = config.fn_aff
        aff_3d = formats.read_aff(infile)
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
    if not outfile:
        outfile = io.get_filename(config.dn_data, "oversegment-%s" % oversegmenter, "tif")
    formats.save_labels(outfile, labels_3d)


def main():
    parser = jargparse.ArgumentParser()
    parser.add_argument('oversegmenter')
    parser.add_argument('-l, --limit', dest="limit", type=int, default=0,
        help="max slices to oversegment")
    parser.add_argument('-i, --infile', dest="infile",
        help="image file to segment, if not supplied then file in config.py will be used")
    parser.add_argument('-o, --outfile', dest="outfile",
        help="labels file to output to, if not supplied then a default path will be used")
    args = parser.parse_args()

    oversegment(args.oversegmenter, args.limit, args.infile, args.outfile)


if __name__ == "__main__":
    args = main()
