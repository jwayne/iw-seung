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
from dtypes import BM_DTYPE, AFF_DTYPE, LABELS_DTYPE, AFF_MAX, AFF_INDEX_MAP


##########
# bm
##########

def read_bm(fn):
    logging.info("Reading bm: '%s'" % fn)
    bm_3d = tifffile.imread(fn)
    # To handle single-page tiffs
    if len(bm_3d.shape) < 3:
        bm_3d = bm_3d.reshape((1,) + bm_3d.shape)
    return bm_3d

def save_bm(fn, bm_3d):
    logging.info("Writing bm: '%s'" % fn)
    if os.path.exists(fn):
        raise IOError("File already exists, will not overwrite: %s" % fn)
    bm_3d = BM_DTYPE(bm_3d)
    tifffile.imsave(fn, bm_3d, photometric='minisblack')


##########
# aff
##########
#TODO: use size 3, instead of size 6, in last dimension

def read_aff(fn):
    logging.info("Reading aff: '%s'" % fn)
    with open(fn, 'rb') as f:
        zsize, ysize, xsize, _ = np.fromfile(f, dtype=np.uint16, count=4)
        aff_3d_f = np.fromfile(f, dtype=AFF_DTYPE)
    aff_3d_f = aff_3d_f.reshape((zsize, ysize, xsize, 3))
    aff_3d = np.zeros((zsize, ysize, xsize, 6), dtype=AFF_DTYPE)
    aff_3d[:, :, :, 0:3] = aff_3d_f
    refresh_aff(aff_3d)
    return aff_3d

def save_aff(fn, aff_3d):
    logging.info("Writing aff: '%s'" % fn)
    if os.path.exists(fn):
        raise IOError("File already exists, will not overwrite: %s" % fn)
    aff_3d_f = aff_3d[:, :, :, :3]
    aff_3d_f = AFF_DTYPE(aff_3d_f)
    with open(fn, 'wb') as f:
        shape = np.uint16(aff_3d_f.shape)
        shape.tofile(f)
        aff_3d_f.tofile(f)

def save_aff_tiff(fn, aff_3d, dim=1):
    """
    Saves affinities going in a single direction to a tiff for easier viewing.
    dim: 0=z, 1=y, 2=x, 3=affv
    """
    logging.info("Writing aff tiff: '%s'" % fn)
    if dim < 3:
        aff_3d_f = aff_3d[:, :, :, dim]
    else:
        aff_3d_f = aff2affv(aff_3d)
    aff_3d_f = AFF_DTYPE(aff_3d_f)
    tifffile.imsave(fn, aff_3d_f, photometric='minisblack')

def refresh_aff(aff_3d):
    aff_3d[1:, :, :, 3] = aff_3d[:-1, :, :, 0]
    aff_3d[:, 1:, :, 4] = aff_3d[:, :-1, :, 1]
    aff_3d[:, :, 1:, 5] = aff_3d[:, :, :-1, 2]

def bm2aff(bm_3d):
    zsize, ysize, xsize = bm_3d.shape
    aff_3d = np.zeros((zsize, ysize, xsize, 6), dtype=AFF_DTYPE)
    # TODO: max or diff?
    aff_3d[:-1, :, :, 0] = np.max((bm_3d[1:, :, :], bm_3d[:-1, :, :]), 0)
    aff_3d[:, :-1, :, 1] = np.max((bm_3d[:, 1:, :], bm_3d[:, :-1, :]), 0)
    aff_3d[:, :, :-1, 2] = np.max((bm_3d[:, :, 1:], bm_3d[:, :, :-1]), 0)
#    aff_3d[:-1, :, :, 0] = AFF_MAX - np.abs(bm_3d[1:, :, :] - bm_3d[:-1, :, :])
#    aff_3d[:, :-1, :, 1] = AFF_MAX - np.abs(bm_3d[:, 1:, :] - bm_3d[:, :-1, :])
#    aff_3d[:, :, :-1, 2] = AFF_MAX - np.abs(bm_3d[:, :, 1:] - bm_3d[:, :, :-1])
    refresh_aff(aff_3d)
    return aff_3d

def aff2affv(aff_3d):
    return np.amax(aff_3d, axis=3)


##########
# labels
##########

def read_labels(fn):
    logging.info("Reading labels: '%s'" % fn)
    labels_3d = tifffile.imread(fn)
    return labels_3d

def save_labels(fn, labels_3d):
    logging.info("Writing labels: '%s'" % fn)
    if os.path.exists(fn):
        raise IOError("File already exists, will not overwrite: %s" % fn)
    labels_3d = LABELS_DTYPE(labels_3d)
    tifffile.imsave(fn, labels_3d, photometric='minisblack')
