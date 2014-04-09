"""
Convert boundary maps (snemi3d) to .aff affinity graphs (snemi3d).
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

bm_3d = formats.read_bm(in_fn)
if lim:
    bm_3d = bm_3d[:lim]
aff_3d = formats.bm2aff(bm_3d)
formats.save_aff(out_fn, aff_3d)
