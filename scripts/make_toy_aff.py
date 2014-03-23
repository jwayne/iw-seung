# figure 4 in cousty 2009
import formats
import numpy as np
import sys

fn = sys.argv[1]
aff_3d = np.zeros((1,4,4,6), dtype=np.uint8)
# Only need to instantiate first 3 edges, since the rest is discarded during saving
# vertical edges
aff_3d[0,:-1,:,1] = 10 - np.array([
    [2, 5, 8, 1],
    [3, 4, 5, 2],
    [3, 4, 7, 0],
])
# horizontal edges
aff_3d[0,:,:-1,2] = 10 - np.array([
    [1, 5, 5],
    [4, 4, 1],
    [6, 5, 0],
    [3, 4, 0],
])
formats.save_aff(fn, aff_3d)
