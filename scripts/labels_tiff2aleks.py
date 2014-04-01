"""
Convert my label output to Aleks's label output format.

For converting output of my algorithms, as well as snemi3d ground truth
labels, to same format as output of Aleks's algorithms.  This is so we
can run the MATLAB metrics.m fn on the output labels.
"""
from jpyutils.jlogging import logging_setup
import numpy as np
import sys
from structs import formats

logging_setup('debug')


def save_labels_aleks(fn, labels_3d):
    labels_3d.tofile(fn)


fn_in = sys.argv[1]
fn_out = sys.argv[2]
if len(sys.argv) > 3:
    lim = int(sys.argv[3])
else:
    lim = 0


labels_3d = formats.read_labels(fn_in)
if lim:
    labels_3d = labels_3d[:lim]
save_labels_aleks(fn_out, labels_3d)
