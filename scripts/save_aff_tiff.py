import config, formats
from jpyutils.jlogging import logging_setup
import sys


logging_setup('debug')
fn = sys.argv[1]
dim = sys.argv[2]
formats.save_aff_tiff(fn, formats.read_aff(config.fn_aff, config.shape_aff), dim)
