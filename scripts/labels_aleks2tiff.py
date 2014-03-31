"""
Convert Aleks' label output to my tiff label output format.
"""
import numpy as np
import sys
from structs import formats

def read_labels_aleks(fn, sz):
    labels_3d = np.fromfile(fn, dtype=formats.LABELS_DTYPE).reshape((sz, sz, sz))
    return labels_3d

fn_in, sz, fn_out = sys.argv[1:4]
labels_3d = read_labels_aleks(fn_in, int(sz))
formats.save_labels(fn_out, labels_3d)
