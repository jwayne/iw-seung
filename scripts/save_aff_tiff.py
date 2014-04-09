"""
Wrapper for formats.save_aff_tiff.
"""
from jpyutils.jlogging import logging_setup
import sys
from structs import formats

logging_setup('debug')


in_fn, out_fn = sys.argv[1:3]
if len(sys.argv) > 3:
    dim = int(sys.argv[3])
else:
    # Default to taking min of all affinities
    dim = 3

formats.save_aff_tiff(out_fn, formats.read_aff(in_fn), dim)
