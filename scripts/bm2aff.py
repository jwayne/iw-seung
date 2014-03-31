"""
Convert bm to aff.
"""
from jpyutils.jlogging import logging_setup
import sys
import config
from structs import formats

logging_setup('debug')


if len(sys.argv) > 1:
    in_fn, out_fn = sys.argv[1:3]
else:
    in_fn = config.fn_bm
    out_fn = config.fn_aff


bm_3d = formats.read_bm(in_fn)
aff_3d = formats.bm2aff(bm_3d)
formats.save_aff(out_fn, aff_3d)
