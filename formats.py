"""
bm:
    3d numpy array [z][x][y] of probability(boundary)
    * elements = 0-255, 1 byte each
    * values = [0, 1=boundary]
aff:
    3d numpy array [z][x][y][3] of edge affinities offset so that
    aff_3d[z][x][y][0,1,2] corresponds to the edge in the +z, +x, +y
    directions of (z,x,y), respectively.
    * elements = 0-255, 1 byte each
    * values = [0, 1=connected]
labels:
    3d numpy array [z][x][y] of integer labels
    * elements = unsigned int, 2 bytes each
    * values = [1, 2, ..]
"""
import logging
import numpy as np
import os
import tifffile


##########
# bm
##########

BM_DTYPE = np.uint8

def read_bm(fn):
    bm_3d = tifffile.imread(fn)
    logging.info("Read bm: '%s'" % fn)
    return bm_3d


##########
# aff
##########

AFF_DTYPE = np.uint8
AFF_MAX = 255

def read_aff(fn, shape):
    aff_3d_f = np.fromfile(fn, dtype=AFF_DTYPE)
    aff_3d_f = aff_3d_f.reshape(shape + (3,))
    logging.info("Read aff: '%s'" % fn)
    aff_3d = np.zeros(shape + (6,), dtype=AFF_DTYPE)
    aff_3d[:, :, :, 0:3] = aff_3d_f
    aff_3d[1:, :, :, 3] = aff_3d_f[:-1, :, :, 0]
    aff_3d[:, 1:, :, 4] = aff_3d_f[:, :-1, :, 1]
    aff_3d[:, :, 1:, 5] = aff_3d_f[:, :, :-1, 2]
    return aff_3d

def save_aff(fn, aff_3d):
    if os.path.exists(fn):
        raise IOError("File already exists, will not overwrite: %s" % fn)
    aff_3d_f = aff_3d[:, :, :, :3]
    aff_3d_f = AFF_DTYPE(aff_3d_f)
    with open(fn, 'w') as f:
        aff_3d_f.tofile(f)
    logging.info("Wrote aff: '%s'" % fn)

def save_aff_tiff(fn, aff_3d, dim=1):
    """
    Saves affinities going in a single direction to a tiff for easier viewing.
    dim: 0=z, 1=x, 2=y, 3=affv
    """
    if dim < 3:
        aff_3d_f = aff_3d[:, :, :, dim]
    else:
        aff_3d_f = aff2affv(aff_3d)
    aff_3d_f = AFF_DTYPE(aff_3d_f)
    tifffile.imsave(fn, aff_3d_f, photometric='minisblack')
    logging.info("Wrote aff tiff: '%s'" % fn)

def bm2aff(bm_3d):
    zsize, xsize, ysize = bm_3d.shape
    aff_3d = np.zeros((zsize, xsize, ysize, 6), dtype=AFF_DTYPE)
    aff_3d[:-1, :, :, 0] = AFF_MAX - np.abs(bm_3d[1:, :, :] - bm_3d[:-1, :, :])
    aff_3d[:, :-1, :, 1] = AFF_MAX - np.abs(bm_3d[:, 1:, :] - bm_3d[:, :-1, :])
    aff_3d[:, :, :-1, 2] = AFF_MAX - np.abs(bm_3d[:, :, 1:] - bm_3d[:, :, :-1])
    aff_3d[1:, :, :, 3] = aff_3d[:-1, :, :, 0]
    aff_3d[:, 1:, :, 4] = aff_3d[:, :-1, :, 1]
    aff_3d[:, :, 1:, 5] = aff_3d[:, :, :-1, 2]
    return aff_3d

def aff2affv(aff_3d):
    return np.amax(aff_3d, axis=3)

AFF_INDEX_MAP = (
    (1,0,0),
    (0,1,0),
    (0,0,1),
    (-1,0,0),
    (0,-1,0),
    (0,0,-1),
)


##########
# labels
##########

LABELS_DTYPE = np.uint16

def read_labels(fn):
    labels_3d = tifffile.imread(fn)
    logging.info("Read labels: '%s'" % fn)
    return labels_3d

def save_labels(fn, labels_3d):
    if os.path.exists(fn):
        raise IOError("File already exists, will not overwrite: %s" % fn)
    labels_3d = LABELS_DTYPE(labels_3d)
    tifffile.imsave(fn, labels_3d, photometric='minisblack')
    logging.info("Wrote labels: '%s'" % fn)
