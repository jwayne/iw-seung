"""
bm:
    3d numpy array [z][x][y] of probability(boundary) [0,1=boundary]
    * elements = 0-255, 1 byte each
aff:
    3d numpy array [z][x][y][3] of edge affinities [0,1=connected],
    offset so that aff_3d[z][x][y][0,1,2] corresponds to the edge
    in the -z, -x, -y directions of (z,x,y), respectively.
    * elements = 0-255, 1 byte each
labels:
    3d numpy array [z][x][y] of integer labels 
    * elements = unsigned int, 2 bytes each
"""
import logging
import numpy as np
import os
import tifffile


##########
# bm
##########

def read_bm(fn):
    bm_3d = tifffile.imread(fn)
    logging.info("Read bm: '%s'" % fn)
    return bm_3d


##########
# aff
##########

def read_aff(fn, xsize, ysize, zsize):
    aff_3d = np.fromfile(fn, dtype=np.uint8)
    aff_3d = aff_3d.reshape((zsize, xsize, ysize, 3))
    logging.info("Read aff: '%s'" % fn)
    return aff_3d

def save_aff(fn, aff_3d):
    if os.path.exists(fn):
        raise IOError("File already exists, will not overwrite: %s" % fn)
    aff_3d = np.uint8(aff_3d)
    with open(fn, 'w') as f:
        aff_3d.tofile(f)
    logging.info("Wrote aff: '%s'" % fn)

def bm2aff(bm_3d):
    zsize, xsize, ysize = bm_3d.shape
    aff_3d = np.zeros((zsize, xsize, ysize, 3))
    aff_3d[1:, :, :, 0] = 255 - np.abs(bm_3d[1:, :, :] - bm_3d[:-1, :, :])
    aff_3d[:, 1:, :, 1] = 255 - np.abs(bm_3d[:, 1:, :] - bm_3d[:, :-1, :])
    aff_3d[:, :, 1:, 2] = 255 - np.abs(bm_3d[:, :, 1:] - bm_3d[:, :, :-1])
    return aff_3d


##########
# labels
##########

def read_labels(fn):
    labels_3d = tifffile.imread(fn)
    logging.info("Read labels: '%s'" % fn)
    return labels_3d

def save_labels(fn, labels_3d):
    if os.path.exists(fn):
        raise IOError("File already exists, will not overwrite: %s" % fn)
    labels_3d = np.uint16(labels_3d)
    tifffile.imsave(fn, labels_3d, photometric='minisblack')
    logging.info("Wrote labels: '%s'" % fn)
