"""
Convert Aleks' label output to my tiff label output format.
"""
import numpy as np
import sys
from structs import formats

def read_labels_aleks(fn, zsize=160, ysize=160, xsize=160):
    labels_3d = np.fromfile(fn, dtype=formats.LABELS_DTYPE).reshape((zsize, ysize, xsize))
    return labels_3d

fn_in, fn_out = sys.argv[1:3]
labels_3d = read_labels_aleks(fn_in)
formats.save_labels(fn_out, labels_3d)
