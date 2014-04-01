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
    Read boundary map values from a tiff file.  dtype is read from the tiff
    file and checked against a list of accepted types.
    """
    if not fn.endswith('.tif'):
        raise ValueError("Bad filename, extension not .tif: %s" % fn)

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
    Save affinities to a tiff file.  dtype is taken from `bm_3d` and
    checked against a list of accepted types.
    """
    if bm_3d.dtype not in (WEIGHT_DTYPE_UINT, WEIGHT_DTYPE_FLOAT):
        raise TypeError("Bad dtype '%s': Must be '%s', '%s'" %
            (bm_3d.dtype, WEIGHT_DTYPE_UINT, WEIGHT_DTYPE_FLOAT))
    if not fn.endswith('.tif'):
        raise ValueError("Bad filename, extension not .tif: %s" % fn)

    logging.info("Writing bm: '%s'" % fn)
    if os.path.exists(fn):
        raise IOError("File already exists, will not overwrite: %s" % fn)
    tifffile.imsave(fn, bm_3d, photometric='minisblack')



##########
# aff
##########
#TODO: use size 3, instead of size 6, in last dimension.  I'm wasting memory..

@timeit
def read_aff(fn, zsize=0, ysize=0, xsize=0):
    """
    Read edge affinities from a binary file.  File format and dtype are inferred
    from `fn`s extension and checked against a list of accepted types.  In
    particular:
    
    If .aff, then the file is assumed to be a sequence of uint8's in
    [z][y][x][+z,+y,+x] order, with the dimensions saved as the first 3 uint16's
    in the binary file.  The 4th uint16 is garbage.
    
    If .raw, then the file is assumed to be a sequence of float32's in
    [-x,-y,-z][z][y][x] order (aleks's data format), with dimensions
    inferred from the filename.  (TODO: dimension inference is hacky)
    """
    ext = fn[-4:]
    if ext == ".aff":
        dtype = WEIGHT_DTYPE_UINT
    elif ext == ".raw":
        dtype = WEIGHT_DTYPE_FLOAT
    else:
        raise ValueError("Bad filename, extension not .aff or .raw: %s" % fn)

    logging.info("Reading aff (%s): '%s'" % (dtype, fn))
    with open(fn, 'rb') as f:
        if ext == ".aff":
            zsize, ysize, xsize, _ = np.fromfile(f, dtype=np.uint16, count=4)
            aff_3d = np.zeros((zsize, ysize, xsize, 6), dtype=dtype)
            aff_3d[:, :, :, :3] = np.fromfile(f, dtype=dtype).reshape((zsize, ysize, xsize, 3))
            refresh_aff(aff_3d)
        elif ext == ".raw":
            if not zsize:
                import re
                sz = int(re.search('(\d+)', os.path.split(fn)[-1]).groups()[0])
                zsize = ysize = xsize = sz
            aff_3d = np.zeros((zsize, ysize, xsize, 6), dtype=dtype)
            per_slice = zsize * ysize * xsize
            for i in [5,4,3]:
                aff_3d[:, :, :, i] = np.fromfile(
                    f, dtype=dtype, count=per_slice).reshape((zsize, ysize, xsize))
            refresh_aff(aff_3d, reverse=True)
        else:
            assert False
    return aff_3d


def save_aff(fn, aff_3d):
    """
    Save edge affinities to a binary file.  File format and dtype are taken
    from `aff_3d` and checked against a list of accepted types.  In particular:

    If .aff, then the file is saved as a sequence of uint8's in
    [z][y][x][+z,+y,+x] order, with the dimensions saved as the first 3 uint16's
    in the binary file.  The 4th uint16 is garbage.
    
    If .raw, then the file is saved as a sequence of float32's in
    [-x,-y,-z][z][y][x] order (aleks's data format), with dimensions
    inferred from the filename.  (TODO: dimension inference is hacky)
    """
    dtype = aff_3d.dtype
    ext = fn[-4:]
    if dtype == WEIGHT_DTYPE_UINT:
        if ext != ".aff":
            raise ValueError("Bad filename, extension not .aff: %s" % fn)
    elif dtype == WEIGHT_DTYPE_FLOAT:
        if ext != ".raw":
            raise ValueError("Bad filename, extension not .raw: %s" % fn)
    else:
        raise TypeError("Bad dtype '%s': Must be '%s' or '%s'" %
            (dtype, WEIGHT_DTYPE_UINT, WEIGHT_DTYPE_FLOAT))

    logging.info("Writing aff (%s): '%s'" % (dtype, fn))
    if os.path.exists(fn):
        raise IOError("File already exists, will not overwrite: %s" % fn)
    with open(fn, 'wb') as f:
        if dtype == WEIGHT_DTYPE_UINT:
            shape = np.uint16(aff_3d.shape)
            shape.tofile(f)
            aff_3d[:,:,:,:3].tofile(f)
        elif dtype == WEIGHT_DTYPE_FLOAT:
            for i in [5,4,3]:
                aff_3d[:,:,:,i].tofile(f)
        else:
            assert False


def save_aff_tiff(fn, aff_3d, dim=3):
    """
    Save edge affinities going in a single direction to a tiff for easier viewing.
    Tiff type is taken from `aff_3d` and checked against a list of accepted types.

    @param dim:
        0=z, 1=y, 2=x, 3=affv
    """
    dtype = aff_3d.dtype
    if dtype != WEIGHT_DTYPE_UINT and dtype != WEIGHT_DTYPE_FLOAT:
        raise TypeError("Bad dtype '%s': Must be '%s' or '%s'" %
            (dtype, WEIGHT_DTYPE_UINT, WEIGHT_DTYPE_FLOAT))
    if not fn.endswith('.tif'):
        raise ValueError("Bad filename, extension not .tif: %s" % fn)

    logging.info("Writing aff tiff (%s): '%s'" % (dtype, fn))
    if os.path.exists(fn):
        raise IOError("File already exists, will not overwrite: %s" % fn)
    if dim < 3:
        aff_3d_f = aff_3d[:, :, :, dim]
    else:
        aff_3d_f = aff2affv(aff_3d)
    tifffile.imsave(fn, aff_3d_f, photometric='minisblack')


def refresh_aff(aff_3d, reverse=False):
    """
    Update the (duplicate) -z, -y, -x directions of the edge affinities array.
    This structure is to enable better cache loading, assuming access are made
    with pixel locality.  Works on all dtypes.

    If reverse=True, then update the +z, +y, +x directions instead.
    """
    if not reverse:
        aff_3d[1:, :, :, 3] = aff_3d[:-1, :, :, 0]
        aff_3d[:, 1:, :, 4] = aff_3d[:, :-1, :, 1]
        aff_3d[:, :, 1:, 5] = aff_3d[:, :, :-1, 2]
    else:
        aff_3d[:-1, :, :, 0] = aff_3d[1:, :, :, 3]
        aff_3d[:, :-1, :, 1] = aff_3d[:, 1:, :, 4]
        aff_3d[:, :, :-1, 2] = aff_3d[:, :, 1:, 5]


def clean_aff(aff_3d):
    """
    To prevent out-of-bounds errors in oversegmenting algorithms, we need to
    have edge affinities be 0.  Works on all dtypes.
    """
    aff_3d[-1, :, :, 0] = 0
    aff_3d[:, -1, :, 1] = 0
    aff_3d[:, :, -1, 2] = 0
    aff_3d[0, :, :, 3] = 0
    aff_3d[:, 0, :, 4] = 0
    aff_3d[:, :, 0, 5] = 0


def bm2aff(bm_3d):
    """
    Convert a boundary map into an affinity graph.  dtype is taken from `bm_3d`
    and checked against a list of accepted types.
    """
    dtype = aff_3d.dtype
    if dtype != WEIGHT_DTYPE_UINT and dtype != WEIGHT_DTYPE_FLOAT:
        raise TypeError("Bad dtype '%s': Must be '%s' or '%s'" %
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
    max weight among all edges entering/leaving the pixel.  Works on all dtypes.
    """
    return np.amax(aff_3d, axis=3)



##########
# labels
##########

def read_labels(fn):
    """
    Read labels from a tiff file.  dtype is taken from the tiff file and
    checked against the accepted type.
    """
    if not fn.endswith('.tif'):
        raise ValueError("Bad extension, not .tif: %s" % fn)

    logging.info("Reading labels: '%s'" % fn)
    labels_3d = tifffile.imread(fn)
    if labels_3d.dtype != LABELS_DTYPE:
        logging.warning("Read dtype '%s', coercing to dtype '%s'" %
            (labels_3d.dtype, LABELS_DTYPE))
        labels_3d = LABELS_DTYPE(labels_3d)
    return labels_3d


def save_labels(fn, labels_3d):
    """
    Write labels to a tiff file.  dtype is taken from `labels_3d` and
    checked against the accepted type.
    """
    if labels_3d.dtype != LABELS_DTYPE:
        raise TypeError("Bad dtype '%s': Must be '%s'" %
            (labels_3d.dtype, LABELS_DTYPE))
    if not fn.endswith('.tif'):
        raise ValueError("Bad extension, not .tif: %s" % fn)

    logging.info("Writing labels: '%s'" % fn)
    if os.path.exists(fn):
        raise IOError("File already exists, will not overwrite: %s" % fn)
    tifffile.imsave(fn, labels_3d, photometric='minisblack')
