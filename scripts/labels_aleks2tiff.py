"""
Convert Aleks' label output to my tiff label output format.

This is for viewing of labels via imagej.
"""
from jpyutils.jlogging import logging_setup
import numpy as np
import sys
from structs import formats

logging_setup('debug')


def read_labels_aleks(fn, zsize, ysize, xsize):
    labels_3d = np.fromfile(fn, dtype=formats.LABELS_DTYPE).reshape((zsize, ysize, xsize))
    return labels_3d


fn_in = sys.argv[1]
zsize, ysize, xsize = map(int, sys.argv[2:5])
fn_out = sys.argv[5]
labels_3d = read_labels_aleks(fn_in, zsize, ysize, xsize)
formats.save_labels(fn_out, labels_3d)
