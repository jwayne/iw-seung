"""
bm:
    3d numpy array [z][y][x] of probability(boundary)
    * elements = 0-255, 1 byte each
    * values = [0, 1=boundary]
aff:
    3d numpy array [z][y][x][6] of edge affinities offset so that
    aff_3d[z][y][x][0,1,2,3,4,5] corresponds to the edge in the
    +z, +y, +x, -z, -y, -x directions of (z,y,x), respectively.
    * elements = 0-255, 1 byte each
    * values = [0, 1=connected]
labels:
    3d numpy array [z][y][x] of integer labels
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

def save_bm(fn, bm_3d):
    if os.path.exists(fn):
        raise IOError("File already exists, will not overwrite: %s" % fn)
    bm_3d = BM_DTYPE(bm_3d)
    tifffile.imsave(fn, bm_3d, photometric='minisblack')
    logging.info("Wrote bm: '%s'" % fn)


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
    refresh_aff(aff_3d)
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
    dim: 0=z, 1=y, 2=x, 3=affv
    """
    if dim < 3:
        aff_3d_f = aff_3d[:, :, :, dim]
    else:
        aff_3d_f = aff2affv(aff_3d)
    aff_3d_f = AFF_DTYPE(aff_3d_f)
    tifffile.imsave(fn, aff_3d_f, photometric='minisblack')
    logging.info("Wrote aff tiff: '%s'" % fn)

def refresh_aff(aff_3d):
    aff_3d[1:, :, :, 3] = aff_3d[:-1, :, :, 0]
    aff_3d[:, 1:, :, 4] = aff_3d[:, :-1, :, 1]
    aff_3d[:, :, 1:, 5] = aff_3d[:, :, :-1, 2]

def bm2aff(bm_3d):
    zsize, ysize, xsize = bm_3d.shape
    aff_3d = np.zeros((zsize, ysize, xsize, 6), dtype=AFF_DTYPE)
    aff_3d[:-1, :, :, 0] = AFF_MAX - np.abs(bm_3d[1:, :, :] - bm_3d[:-1, :, :])
    aff_3d[:, :-1, :, 1] = AFF_MAX - np.abs(bm_3d[:, 1:, :] - bm_3d[:, :-1, :])
    aff_3d[:, :, :-1, 2] = AFF_MAX - np.abs(bm_3d[:, :, 1:] - bm_3d[:, :, :-1])
    refresh_aff(aff_3d)
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
