from jpyutils.jlogging import logging_setup
import sys
import config
from structs import formats

logging_setup('debug')


if len(sys.argv) > 3:
    in_fn = sys.argv[1]
    dim = sys.argv[2]
    out_fn = sys.argv[3]
else:
    in_fn = config.fn_aff
    dim = sys.argv[1]
    out_fn = sys.argv[2]

formats.save_aff_tiff(out_fn, formats.read_aff(in_fn), dim)
