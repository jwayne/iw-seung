"""
Convert boundary maps (snemi3d) to .aff affinity graphs (snemi3d).
"""
from jpyutils.jlogging import logging_setup
import sys
from structs import formats

logging_setup('debug')


in_fn, out_fn = sys.argv[1:3]


bm_3d = formats.read_bm(in_fn)
aff_3d = formats.bm2aff(bm_3d)
formats.save_aff(out_fn, aff_3d)
