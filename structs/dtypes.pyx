# cython: profile = False
# distutils: language = c++
import numpy as np
cimport numpy as np


WEIGHT_DTYPE_UINT = np.uint8
WEIGHT_DTYPE_FLOAT = np.float32
ctypedef np.uint8_t WEIGHT_DTYPE_UINT_t
ctypedef np.float32_t WEIGHT_DTYPE_FLOAT_t
ctypedef fused WEIGHT_DTYPE_t:
    np.uint8_t
    np.float32_t
WEIGHT_MAX_UINT = 255

LABELS_DTYPE = np.uint32
ctypedef np.uint32_t LABELS_DTYPE_t


AFF_INDEX_MAP = np.array((
    (1,0,0),
    (0,1,0),
    (0,0,1),
    (-1,0,0),
    (0,-1,0),
    (0,0,-1),
), dtype=np.int)
cdef np.int_t[:,:] AFF_INDEX_MAP_c = AFF_INDEX_MAP
