"""
Convert edge affinities file from .aff (snemi3d) to .raw (aleks).
"""
from jpyutils.jlogging import logging_setup
import sys
from structs import formats

logging_setup('debug')


in_fn, out_fn = sys.argv[1:3]
if len(sys.argv) > 3:
    lim = int(sys.argv[3])
else:
    lim = 0


aff_3d = formats.read_aff(in_fn)
if lim:
    aff_3d = aff_3d[:lim]
aff_3d = formats.WEIGHT_DTYPE_FLOAT(aff_3d) / formats.WEIGHT_MAX_UINT
formats.save_aff(out_fn, aff_3d)
