import numpy as np
cimport numpy as np

BM_DTYPE = np.uint8
ctypedef np.uint8_t BM_DTYPE_t

AFF_DTYPE = np.uint8
ctypedef np.uint8_t AFF_DTYPE_t
# not used in any cython stuff so leaving it as a python object is ok
AFF_MAX = 255

LABELS_DTYPE = np.uint16
ctypedef np.uint16_t LABELS_DTYPE_t

AFF_INDEX_MAP = np.array((
    (1,0,0),
    (0,1,0),
    (0,0,1),
    (-1,0,0),
    (0,-1,0),
    (0,0,-1),
), dtype=np.int)
cdef np.int_t[:,:] AFF_INDEX_MAP_c = AFF_INDEX_MAP
