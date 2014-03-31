"""
Convert Aleks' label output to my tiff label output format. (reversed of this)
"""
import numpy as np
import sys
from structs import formats

def save_labels_aleks(fn, labels_3d):
    labels_3d.tofile(fn)

fn_in = sys.argv[1]
fn_out = sys.argv[2]
labels_3d = formats.read_labels(fn_in)
if len(sys.argv) > 3:
    lim = int(sys.argv[3])
    labels_3d = labels_3d[:lim]
save_labels_aleks(fn_out, labels_3d)
