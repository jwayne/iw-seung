"""
bm: pixel membrane probabilities
    3d numpy array [z][y][x] of probability(boundary)
    * elements = 0-255, 1 byte each
    * values = [0, 1=boundary]
aff: edge weights
    3d numpy array [z][y][x][6] of edge affinities offset so that
    aff_3d[z][y][x][0,1,2,3,4,5] corresponds to the edge in the
    +z, +y, +x, -z, -y, -x directions of (z,y,x), respectively.
    * elements = 0-255, 1 byte each
    * values = [0, 1=connected]
labels: pixel labels
    3d numpy array [z][y][x] of integer labels
    * elements = unsigned int, 2 bytes each
    * values = [1, 2, ..]
"""
from jpyutils.timeit import timeit
import logging
import numpy as np
import os
import tifffile
from structs.dtypes import (WEIGHT_DTYPE_UINT, WEIGHT_DTYPE_FLOAT, LABELS_DTYPE,
                            WEIGHT_MAX_UINT, AFF_INDEX_MAP)


##########
# bm
##########

def read_bm(fn):
    """
    Read boundary map values from a tiff file.
    Type is inferred from the tiff file and checked against a list of accepted types.
    """
    if not fn.endswith('.tif'):
        raise ValueError("Bad extension, not .tif: %s" % fn)

    logging.info("Reading bm: '%s'" % fn)
    bm_3d = tifffile.imread(fn)
    if bm_3d.dtype not in (WEIGHT_DTYPE_UINT, WEIGHT_DTYPE_FLOAT):
        raise TypeError("Bad dtype '%s': Must be '%s', '%s'" %
            (bm_3d.dtype, WEIGHT_DTYPE_UINT, WEIGHT_DTYPE_FLOAT))
    # To handle single-page tiffs
    if len(bm_3d.shape) < 3:
        bm_3d = bm_3d.reshape((1,) + bm_3d.shape)
    return bm_3d


def save_bm(fn, bm_3d):
    """
    Save affinities to a tiff file.
    Type is inferred from `bm_3d` and checked against a list of accepted types.
    """
    if not fn.endswith('.tif'):
        raise ValueError("Bad extension, not .tif: %s" % fn)
    if bm_3d.dtype not in (WEIGHT_DTYPE_UINT, WEIGHT_DTYPE_FLOAT):
        raise TypeError("Bad dtype '%s': Must be '%s', '%s'" %
            (bm_3d.dtype, WEIGHT_DTYPE_UINT, WEIGHT_DTYPE_FLOAT))

    logging.info("Writing bm: '%s'" % fn)
    if os.path.exists(fn):
        raise IOError("File already exists, will not overwrite: %s" % fn)
    tifffile.imsave(fn, bm_3d, photometric='minisblack')



##########
# aff
##########
#TODO: use size 3, instead of size 6, in last dimension.  I'm wasting memory..

@timeit
def read_aff(fn):
    """
    Read edge affinities from a raw binary file, saved in [z][y][x][dim] order.
    Type is inferred from `fn`s extension.  If .raw, then sizes are assumed to be
    aleks's benchmark dataset.  (If .aff, then sizes are saved in the file.)
    """
    if fn.endswith('.aff'):
        dtype = WEIGHT_DTYPE_UINT
    elif fn.endswith('.raw'):
        dtype = WEIGHT_DTYPE_FLOAT
    else:
        raise ValueError("Bad extension, not .aff or .raw: %s" % fn)

    logging.info("Reading aff (%s): '%s'" % (dtype.__name__, fn))
    with open(fn, 'rb') as f:
        if dtype == WEIGHT_DTYPE_UINT:
            zsize, ysize, xsize, _ = np.fromfile(f, dtype=np.uint16, count=4)
            aff_3d = np.zeros((zsize, ysize, xsize, 6), dtype=dtype)
            aff_3d[:, :, :, :3] = np.fromfile(f, dtype=dtype).reshape((zsize, ysize, xsize, 3))
        elif dtype == WEIGHT_DTYPE_FLOAT:
            import re
            sz = int(re.match('ws_test_(\d+).raw', os.path.split(fn)[-1]).groups()[0])
            aff_3d = np.zeros((sz, sz, sz, 6), dtype=dtype)
            per_slice = sz ** 3
            for i in xrange(3):
                aff_3d[:, :, :, i] = np.fromfile(
                    f, dtype=dtype, count=per_slice).reshape((sz, sz, sz))
        else:
            assert False
    refresh_aff(aff_3d)
    return aff_3d


def save_aff(fn, aff_3d):
    """
    Save edge affinities to a raw binary file, saved in [z][y][x][dim] order.
    Type is inferred from `fn`s extension, and is checked against `aff_3d`s
    dtype and a list of accepted types.
    """
    dtype = aff_3d.dtype
    if dtype == WEIGHT_DTYPE_UINT:
        if not fn.endswith('.aff'):
            raise ValueError("Bad extension, not .aff: %s" % fn)
    elif dtype == WEIGHT_DTYPE_FLOAT:
        if not fn.endswith('.raw'):
            raise ValueError("Bad extension, not .raw: %s" % fn)
    else:
        raise TypeError("Bad dtype '%s': Must be '%s', '%s'" %
            (dtype, WEIGHT_DTYPE_UINT, WEIGHT_DTYPE_FLOAT))

    logging.info("Writing aff (%s): '%s'" % (dtype.__name__, fn))
    if os.path.exists(fn):
        raise IOError("File already exists, will not overwrite: %s" % fn)
    aff_3d_f = aff_3d[:, :, :, :3]
    with open(fn, 'wb') as f:
        if dtype == WEIGHT_DTYPE_UINT:
            shape = np.uint16(aff_3d_f.shape)
            shape.tofile(f)
            aff_3d_f.tofile(f)
        elif dtype == WEIGHT_DTYPE_FLOAT:
            for i in xrange(3):
                aff_3d_f[:,:,:,i].tofile(f)
        else:
            assert False


def save_aff_tiff(fn, aff_3d, dim=3):
    """
    Save edge affinities going in a single direction to a tiff for easier viewing.
    Type is inferred from `aff_3d` and checked against a list of accepted types.

    @param dim:
        0=z, 1=y, 2=x, 3=affv
    """
    dtype = aff_3d.dtype
    if dtype != WEIGHT_DTYPE_UINT and dtype != WEIGHT_DTYPE_FLOAT:
        raise TypeError("Bad dtype '%s': Must be '%s', '%s'" %
            (dtype, WEIGHT_DTYPE_UINT, WEIGHT_DTYPE_FLOAT))
    if not fn.endswith('.tif'):
        raise ValueError("Bad extension, not .tif: %s" % fn)

    logging.info("Writing aff tiff (%s): '%s'" % (dtype, fn))
    if os.path.exists(fn):
        raise IOError("File already exists, will not overwrite: %s" % fn)
    if dim < 3:
        aff_3d_f = aff_3d[:, :, :, dim]
    else:
        aff_3d_f = aff2affv(aff_3d)
    tifffile.imsave(fn, aff_3d_f, photometric='minisblack')


def refresh_aff(aff_3d):
    """
    Update the (duplicate) -z, -y, -x directions of the edge affinities array.
    This structure is to enable better cache loading, assuming access are made
    with pixel locality.
    """
    aff_3d[1:, :, :, 3] = aff_3d[:-1, :, :, 0]
    aff_3d[:, 1:, :, 4] = aff_3d[:, :-1, :, 1]
    aff_3d[:, :, 1:, 5] = aff_3d[:, :, :-1, 2]


def clean_aff(aff_3d):
    """
    To prevent out-of-bounds errors in oversegmenting algorithms, we need to
    have edge affinities be 0.
    """
    aff_3d[-1, :, :, 0] = 0
    aff_3d[:, -1, :, 1] = 0
    aff_3d[:, :, -1, 2] = 0
    aff_3d[0, :, :, 3] = 0
    aff_3d[:, 0, :, 4] = 0
    aff_3d[:, :, 0, 5] = 0


def bm2aff(bm_3d):
    """
    Convert a boundary map into an affinity graph.
    Type is inferred from `bm_3d` and checked against a list of accepted types.
    """
    dtype = aff_3d.dtype
    if dtype != WEIGHT_DTYPE_UINT and dtype != WEIGHT_DTYPE_FLOAT:
        raise TypeError("Bad dtype '%s': Must be '%s', '%s'" %
            (dtype, WEIGHT_DTYPE_UINT, WEIGHT_DTYPE_FLOAT))

    zsize, ysize, xsize = bm_3d.shape
    aff_3d = np.zeros((zsize, ysize, xsize, 6), dtype=dtype)
    # Set affinity of x->y to max(x, y).
    aff_3d[:-1, :, :, 0] = np.max((bm_3d[1:, :, :], bm_3d[:-1, :, :]), 0)
    aff_3d[:, :-1, :, 1] = np.max((bm_3d[:, 1:, :], bm_3d[:, :-1, :]), 0)
    aff_3d[:, :, :-1, 2] = np.max((bm_3d[:, :, 1:], bm_3d[:, :, :-1]), 0)
#    aff_3d[:-1, :, :, 0] = WEIGHT_MAX_UINT - np.abs(bm_3d[1:, :, :] - bm_3d[:-1, :, :])
#    aff_3d[:, :-1, :, 1] = WEIGHT_MAX_UINT - np.abs(bm_3d[:, 1:, :] - bm_3d[:, :-1, :])
#    aff_3d[:, :, :-1, 2] = WEIGHT_MAX_UINT - np.abs(bm_3d[:, :, 1:] - bm_3d[:, :, :-1])
    refresh_aff(aff_3d)
    return aff_3d


def aff2affv(aff_3d):
    """
    Convert an affinity graph into a pixel mapping, mapping each pixel to the
    max weight among all edges entering/leaving the pixel.
    """
    return np.amax(aff_3d, axis=3)



##########
# labels
##########

def read_labels(fn):
    """
    Read labels from a tiff file.
    Type is inferred from tiff file and checked against the accepted type.
    """
    if not fn.endswith('.tif'):
        raise ValueError("Bad extension, not .tif: %s" % fn)

    logging.info("Reading labels: '%s'" % fn)
    labels_3d = tifffile.imread(fn)
    if labels_3d.dtype != LABELS_DTYPE:
        raise TypeError("Bad dtype '%s': Must be '%s'" %
            (labels_3d.dtype, LABELS_DTYPE))
    return labels_3d

def save_labels(fn, labels_3d):
    """
    Write labels to a tiff file.
    Type is inferred from `labels_3d` and checked against the accepted type.
    """
    if not fn.endswith('.tif'):
        raise ValueError("Bad extension, not .tif: %s" % fn)
    if labels_3d.dtype != LABELS_DTYPE:
        raise TypeError("Bad dtype '%s': Must be '%s'" %
            (labels_3d.dtype, LABELS_DTYPE))

    logging.info("Writing labels: '%s'" % fn)
    if os.path.exists(fn):
        raise IOError("File already exists, will not overwrite: %s" % fn)
    tifffile.imsave(fn, labels_3d, photometric='minisblack')
