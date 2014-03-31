"""
Convert aff to raw.
"""
from jpyutils.jlogging import logging_setup
import sys
import config
from structs import formats

logging_setup('debug')


if len(sys.argv) > 1:
    in_fn, out_fn = sys.argv[1:3]
else:
    in_fn = config.fn_aff
    out_fn = config.fn_aff[:-3] + 'raw'


aff_3d = formats.read_aff(in_fn)
aff_3d = formats.WEIGHT_DTYPE_FLOAT(aff_3d[:25]) / formats.WEIGHT_MAX_UINT
formats.save_aff(out_fn, aff_3d)
