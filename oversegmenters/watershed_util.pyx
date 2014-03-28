# cython: profile = True
# distutils: language = c++
"""
Implements:

Cousty et al. 2009 - Watershed Cuts: Minimum Spanning Forests and the Drop of
Water Principle
http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4564470
"""
import logging
import numpy as np
cimport cython
from libcpp.queue cimport queue
from jpyutils.structs.unionfind import UnionFind
from jpyutils.timeit import timeit

from structs import formats
include "structs/dtypes.pyx"


#TODO: danger that n_labels overflows
#TODO: sizes can probably be made a C/C++ object.  but might overflow
#TODO: T_s is unsigned int.  Might overflow


@timeit
@cython.boundscheck(False)
@cython.wraparound(False)
def connected_components(
        AFF_DTYPE_t[:,:,:,:] aff,
        AFF_DTYPE_t[:,:,:] affv,
        AFF_DTYPE_t T_h,
        LABELS_DTYPE_t[:,:,:] labels,
        LABELS_DTYPE_t n_labels = 0,
        object sizes = None):
    """
    @param aff:
        z*y*x*6 edge weight array, containing inter-pixel edge affinities
    @param affv:
        z*y*x edge weight array, containing the max weight of all edges adjacent to each pixel
    @param T_h:
        all edges >= T_h are considered connected
    @param labels:
        z*y*x label array, for each pixel
    @param n_labels:
        number of segments already assigned.  new segments are assigned starting with
        labels of n_labels+1.
    @param sizes:
        defaultdict(int) of segment sizes for each segment
    @return:
        n_labels after this step.
        (note) labels, sizes are updated.
    """
    # Otherwise, we need bounds checking in the loop.
    assert T_h > 0

    cdef LABELS_DTYPE_t n_labels_0 = n_labels
    cdef unsigned int zsize = aff.shape[0]
    cdef unsigned int ysize = aff.shape[1]
    cdef unsigned int xsize = aff.shape[2]
    cdef unsigned int z, y, x, z1, y1, x1, i

    cdef queue[unsigned int] qz, qy, qx

    for z in xrange(zsize):
        for y in xrange(ysize):
            for x in xrange(xsize):
                if not labels[z,y,x] and affv[z,y,x] >= T_h:
                    n_labels += 1
                    labels[z,y,x] = n_labels
                    if sizes is not None:
                        sizes[n_labels] += 1
                    qz.push(z)
                    qy.push(y)
                    qx.push(x)
                    while not qz.empty():
                        z = qz.front()
                        y = qy.front()
                        x = qx.front()
                        qz.pop()
                        qy.pop()
                        qx.pop()
                        # Only need to check 3 directions since otherwise each edge
                        # would be checked twice.
                        for i in xrange(3):
                            if aff[z,y,x,i] >= T_h:
                                # Check if unlabeled
                                z1 = <unsigned int>(z + AFF_INDEX_MAP_c[i][0])
                                y1 = <unsigned int>(y + AFF_INDEX_MAP_c[i][1])
                                x1 = <unsigned int>(x + AFF_INDEX_MAP_c[i][2])
                                if not labels[z1,y1,x1]:
                                    # Note: labels also functions as explored
                                    labels[z1,y1,x1] = n_labels
                                    if sizes:
                                        sizes[n_labels] += 1
                                    qz.push(z1)
                                    qy.push(y1)
                                    qx.push(x1)

    logging.debug("connected_components:(.., T_h=%s, ..): %d new labels found"
        % (T_h, n_labels - n_labels_0))
    return n_labels


@timeit
@cython.boundscheck(False)
@cython.wraparound(False)
def watershed(
        AFF_DTYPE_t[:,:,:,:] aff,
        AFF_DTYPE_t[:,:,:] affv,
        AFF_DTYPE_t T_l,
        LABELS_DTYPE_t[:,:,:] labels,
        LABELS_DTYPE_t n_labels = 0,
        object sizes = None):
    """
    @param aff:
        z*y*x*6 edge weight array, containing inter-pixel edge affinities
    @param affv:
        z*y*x edge weight array, containing the max weight of all edges adjacent to each pixel
    @param T_l:
        no edges < T_l are followed
    @param labels:
        z*y*x label array, for each pixel
    @param n_labels:
        number of segments already assigned.  new segments are assigned starting with
        labels of n_labels+1.
    @param sizes:
        defaultdict(int) of segment sizes for each segment
    @return:
        n_labels after this step.
        (note) labels, sizes are updated.
    """
    # Otherwise, we need bounds checking in the loop.
    assert T_l > 0

    cdef LABELS_DTYPE_t n_labels_0 = n_labels
    cdef unsigned int zsize = aff.shape[0]
    cdef unsigned int ysize = aff.shape[1]
    cdef unsigned int xsize = aff.shape[2]
    cdef unsigned int z, y, x, z1, y1, x1, i
    cdef LABELS_DTYPE_t label

    cdef queue[unsigned int] qz, qy, qx
    #TODO: optimize explored

    for z in xrange(zsize):
        for y in xrange(ysize):
            for x in xrange(xsize):
                if not labels[z,y,x] and affv[z,y,x] >= T_l:
                    qz.push(z)
                    qy.push(y)
                    qx.push(x)
                    explored = set([(z,y,x)])
                    label = 0
                    while not qz.empty():
                        z = qz.front()
                        y = qy.front()
                        x = qx.front()
                        qz.pop()
                        qy.pop()
                        qx.pop()
                        # Need to check 6 directions since otherwise certain 'under'
                        # directions would be missed.
                        for i in xrange(6):
                            if aff[z,y,x,i] >= T_l and aff[z,y,x,i] == affv[z,y,x]:
                                # Check if unlabeled
                                z1 = <unsigned int>(z + AFF_INDEX_MAP_c[i][0])
                                y1 = <unsigned int>(y + AFF_INDEX_MAP_c[i][1])
                                x1 = <unsigned int>(x + AFF_INDEX_MAP_c[i][2])
                                if (z1,y1,x1) in explored:
                                    continue
                                if labels[z1,y1,x1]:
                                    # Found inf-stream under this stream, so 
                                    label = labels[z1,y1,x1]
                                    while not qz.empty():
                                        qz.pop()
                                        qy.pop()
                                        qx.pop()
                                    break
                                explored.add((z1,y1,x1))
                                if affv[z1,y1,x1] > affv[z,y,x]:
                                    # Found new bottom for this stream, so replace q
                                    # TODO: explore if taking max of all matches here affects result?
                                    while not qz.empty():
                                        qz.pop()
                                        qy.pop()
                                        qx.pop()
                                    qz.push(z1)
                                    qy.push(y1)
                                    qx.push(x1)
                                    break
                                # Otherwise, affv[z1,y1,x1] == affv[z,y,x], so we found an
                                # equivalent possible bottom for this stream, so augment q
                                qz.push(z1)
                                qy.push(y1)
                                qx.push(x1)
                    if not label:
                        n_labels += 1
                        label = n_labels
                    for z1,y1,x1 in explored:
                        labels[z1,y1,x1] = label
                    if sizes:
                        sizes[label] += len(explored)
    logging.debug("watershed(.., T_l=%s, ..): %d new labels found"
       % (T_l, n_labels - n_labels_0))
    return n_labels


@timeit
@cython.boundscheck(False)
@cython.wraparound(False)
def get_region_graph(
        AFF_DTYPE_t[:,:,:,:] aff,
        LABELS_DTYPE_t[:,:,:] labels,
        LABELS_DTYPE_t n_labels):
    """
    Create a list of edges connecting the segments in labels.
    @return:
        Region graph of (affinity, label1, label2)
        in decreasing order of affinities.
    """
    assert n_labels >= 3

    cdef unsigned int zsize = aff.shape[0]
    cdef unsigned int ysize = aff.shape[1]
    cdef unsigned int xsize = aff.shape[2]
    cdef unsigned int z, y, x, z1, y1, x1, i, ind
    cdef AFF_DTYPE_t f
    cdef LABELS_DTYPE_t s0, s1, s2

    # compact format of the adjacency graph, where the weight between segments
    # s0,s1 is region_graph[(s0-2)*(s0-1)/2 + (s1-1)]
    cdef unsigned int N = <unsigned int>((n_labels-2)*(n_labels+1)//2)
    cdef AFF_DTYPE_t[::1] aff_segments = np.zeros(N, dtype=AFF_DTYPE)

    # Compute the max edge weight straddling each pair of segments.
    for z in xrange(zsize):
        for y in xrange(ysize):
            for x in xrange(xsize):
                s0 = labels[z,y,x]
                if s0:
                    # Only need to check 3 directions since otherwise each edge
                    # would be checked twice.
                    for i in xrange(3):
                        f = aff[z,y,x,i]
                        if f:
                            z1 = <unsigned int>(z + AFF_INDEX_MAP_c[i][0])
                            y1 = <unsigned int>(y + AFF_INDEX_MAP_c[i][1])
                            x1 = <unsigned int>(x + AFF_INDEX_MAP_c[i][2])
                            s1 = labels[z1,y1,x1]
                            if s1 and s1 != s0:
                                if s0 < s1:
                                    s2 = s1
                                    s1 = s0
                                    s0 = s2
                                ind = <unsigned int>((s0-2)*(s0-1)//2+s1-1)
                                aff_segments[ind] = max(aff_segments[ind], f)

    # Create the sorted (descending) list of edge weights
    #TODO: optimize region_graph
    region_graph = []
    for s0 in xrange(2, n_labels+1):
        for s1 in xrange(1, s0):
            ind = <unsigned int>((s0-2)*(s0-1)//2+s1-1)
            region_graph.append((aff_segments[ind], s0, s1))
    region_graph = sorted(region_graph, reverse=True)

    logging.debug("region_graph(...) completed")
    return region_graph


@timeit
def merge_segments(
        object region_graph,
        LABELS_DTYPE_t[:,:,:] labels,
        LABELS_DTYPE_t n_labels,
        AFF_DTYPE_t T_e,
        unsigned int T_s,
        object sizes):
    """
    @return:
        n_labels after this step.
        (note) labels are updated.
        (note) region_graph, sizes are NOT updated.  This was deemed
        unimportant as merge_segments isn't expected to be called recursively.
    """
    cdef LABELS_DTYPE_t n_labels_0 = n_labels
    cdef unsigned int zsize = labels.shape[0]
    cdef unsigned int ysize = labels.shape[1]
    cdef unsigned int xsize = labels.shape[2]
    cdef unsigned int z, y, x
    cdef AFF_DTYPE_t f
    cdef LABELS_DTYPE_t s0, s1, s0_root, s1_root, s2_root

    #TODO: optimize uf
    uf = UnionFind()
    uf.insert_objects(xrange(1, n_labels+1))
    for f, s0, s1 in region_graph:
        s0_root = uf.find(s0)
        s1_root = uf.find(s1)
        if s0_root != s1_root and f >= T_e:
            if sizes[s0_root] <= T_s or sizes[s1_root] <= T_s:
                s2_root = uf.union(s0_root, s1_root)
                sizes[s2_root] = sizes[s0_root] + sizes[s1_root]
                n_labels -= 1

    # TODO: Sanity check that n_labels has been decremented properly?
    n_labels = len(uf)
    # Map from old labels to new
    # TODO: optimize this?
    label_map = dict((root,i+1) for i,root in enumerate(uf.get_roots()))

    # Discard all segments of size < T_s.
    # TODO: optimize this?
    sizes2 = dict((new_label, sizes[root]) for root,new_label in label_map.iteritems())
    for label, size in sizes2.iteritems():
        if size < T_s:
            label_map[label] = 0
    # TODO: update sizes (return?)

    # Update `labels` with new labels of each pixel
    for z in xrange(zsize):
        for y in xrange(ysize):
            for x in xrange(xsize):
                labels[z,y,x] = label_map[uf.find(labels[z,y,x])]

    # TODO: update region_graph (return?)

    logging.debug("merge_segments(.., T_e=%s, T_s=%s, ..): %d merges made"
        % (T_e, T_s, n_labels_0 - n_labels))
    return n_labels
