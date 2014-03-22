from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import os
from jpyutils import importer, io, jargparse
import tifffile



def plot_oversegmented(arr_labels):
    plot_arr(arr_labels, plt.cm.spectral)
    plt.show()

def plot_arr(arr, cmap):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(arr, cmap=cmap, interpolation='nearest')



def main(oversegmenter, dir_data, to_plot=False, lim=0):

    oversegment = importer.get_module('oversegmenters', oversegmenter).oversegment

    # Open stack of slices stored as tiff
    stack_bm = tifffile.imread(os.path.join(dir_data, 'train-membranes-idsia.tif'))
    if lim:
        stack_bm = stack_bm[:lim]

    stack_labels, n_labels = oversegment(stack_bm)
    stack_labels = np.uint16( stack_labels ) 

    if to_plot:
        for slice_labels in stack_labels:
            plot_oversegmented(slice_labels)

    # Write labels to disk
    fn = io.get_filename(dir_data, "oversegment-%s" % oversegmenter, "tif")
    tifffile.imsave(fn, stack_labels, photometric='minisblack')



if __name__ == "__main__":
    parser = jargparse.ArgumentParser()
    parser.add_argument('oversegmenter')
    parser.add_argument('dir_data')
    parser.add_argument('--lim', type=int, default=0,
        help="max slices to oversegment")
    parser.add_argument('--plot', action='store_true',
        help="plot oversegmented labels for each slice")
    args = parser.parse_args()

    main(args.oversegmenter, args.dir_data, args.plot, args.lim)
