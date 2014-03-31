# cython: profile = True
# distutils: language = c++
"""
Implements:

Cousty et al. 2009 - Watershed Cuts: Minimum Spanning Forests and the Drop of
Water Principle
http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4564470
"""
from jpyutils.structs.unionfind import UnionFind
from jpyutils.timeit import timeit
import logging
import numpy as np
from operator import itemgetter

cimport cython
from libcpp.queue cimport queue

from structs import formats
include "structs/dtypes.pyx"

cdef inline WEIGHT_DTYPE_t aff_max(WEIGHT_DTYPE_t a, WEIGHT_DTYPE_t b): return a if a >= b else b


#TODO: danger that n_labels overflows
#TODO: sizes can probably be made a C/C++ object.  but might overflow
#TODO: T_s is unsigned int.  Might overflow


@timeit
@cython.boundscheck(False)
@cython.wraparound(False)
def connected_components(
        WEIGHT_DTYPE_t[:,:,:,:] aff,
        WEIGHT_DTYPE_t[:,:,:] affv,
        WEIGHT_DTYPE_t T_h,
        LABELS_DTYPE_t[:,:,:] labels,
        LABELS_DTYPE_t n_labels = 0,
        dict sizes = None):
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
        dict of segment sizes for each segment
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
    cdef unsigned int z, y, x, z1, y1, x1, z2, y2, x2, i

    cdef queue[unsigned int] qz, qy, qx

    for z in xrange(zsize):
        for y in xrange(ysize):
            for x in xrange(xsize):
                if not labels[z,y,x] and affv[z,y,x] >= T_h:
                    n_labels += 1
                    labels[z,y,x] = n_labels
                    if sizes is not None:
                        if n_labels in sizes:
                            sizes[n_labels] += 1
                        else:
                            sizes[n_labels] = 1
                    qz.push(z)
                    qy.push(y)
                    qx.push(x)
                    while not qz.empty():
                        z1 = qz.front()
                        y1 = qy.front()
                        x1 = qx.front()
                        qz.pop()
                        qy.pop()
                        qx.pop()
                        # Explore all neighbors.
                        for i in xrange(6):
                            if aff[z1,y1,x1,i] >= T_h:
                                # Note: Guaranteed not out-of-bounds thanks to T_h.
                                z2 = <unsigned int>(z1 + AFF_INDEX_MAP_c[i][0])
                                y2 = <unsigned int>(y1 + AFF_INDEX_MAP_c[i][1])
                                x2 = <unsigned int>(x1 + AFF_INDEX_MAP_c[i][2])
                                # Note: labels[] also functions as the set of
                                # explored pixels in this bfs, so we ensure we don't
                                # explore any pixel twice via this next check.
                                if not labels[z2,y2,x2]:
                                    labels[z2,y2,x2] = n_labels
                                    # sizes[n_labels] guaranteeed to exist.
                                    if sizes:
                                        sizes[n_labels] += 1
                                    qz.push(z2)
                                    qy.push(y2)
                                    qx.push(x2)
                                else:
                                    assert labels[z2,y2,x2] == n_labels

    logging.debug("connected_components:(.., T_h=%s, ..): %d new labels found"
        % (T_h, n_labels - n_labels_0))
    return n_labels


@timeit
@cython.boundscheck(False)
@cython.wraparound(False)
def watershed(
        WEIGHT_DTYPE_t[:,:,:,:] aff,
        WEIGHT_DTYPE_t[:,:,:] affv,
        WEIGHT_DTYPE_t T_l,
        LABELS_DTYPE_t[:,:,:] labels,
        LABELS_DTYPE_t n_labels = 0,
        dict sizes = None):
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
        dict of segment sizes for each segment
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
    cdef unsigned int z, y, x, z1, y1, x1, z2, y2, x2, i
    cdef LABELS_DTYPE_t label

    cdef queue[unsigned int] qz, qy, qx
    #TODO: optimize explored
    cdef set explored

    for z in xrange(zsize):
        for y in xrange(ysize):
            for x in xrange(xsize):
                if not labels[z,y,x] and affv[z,y,x] >= T_l:
                    qz.push(z)
                    qy.push(y)
                    qx.push(x)
                    explored = set([(z,y,x)])
                    label = 0
                    # Follow stream until finding (1) a inf-stream under this stream,
                    # or (2) that this stream is an inf-stream.  If (1), then use
                    # that existing label.  If (2), which happens when q becomes
                    # empty, then create a new label.
                    while not qz.empty():
                        z1 = qz.front()
                        y1 = qy.front()
                        x1 = qx.front()
                        qz.pop()
                        qy.pop()
                        qx.pop()
                        # Explore the neighbor of steepest descent.
                        for i in xrange(6):
                            if aff[z1,y1,x1,i] >= T_l and aff[z1,y1,x1,i] == affv[z1,y1,x1]:
                                # Note: Guaranteed not out-of-bounds thanks to T_l.
                                z2 = <unsigned int>(z1 + AFF_INDEX_MAP_c[i][0])
                                y2 = <unsigned int>(y1 + AFF_INDEX_MAP_c[i][1])
                                x2 = <unsigned int>(x1 + AFF_INDEX_MAP_c[i][2])
                                if (z2,y2,x2) in explored:
                                    continue
                                if labels[z2,y2,x2]:
                                    # (1) Found inf-stream under this stream
                                    label = labels[z2,y2,x2]
                                    # Empty out 'q' so the while loop also ends
                                    while not qz.empty():
                                        qz.pop()
                                        qy.pop()
                                        qx.pop()
                                    break
                                explored.add((z2,y2,x2))
                                if affv[z2,y2,x2] > affv[z1,y1,x1]:
                                    # Found new bottom for this stream, so replace q
                                    #TODO: explore if taking max of all matches here affects result?
                                    while not qz.empty():
                                        qz.pop()
                                        qy.pop()
                                        qx.pop()
                                    qz.push(z2)
                                    qy.push(y2)
                                    qx.push(x2)
                                    break
                                # Otherwise, affv[z2,y2,x2] == affv[z1,y1,x1], so we found an
                                # equivalent possible bottom for this stream, so augment q
                                qz.push(z2)
                                qy.push(y2)
                                qx.push(x2)
                    if not label:
                        n_labels += 1
                        label = n_labels
                    for z1,y1,x1 in explored:
                        labels[z1,y1,x1] = label
                    if sizes is not None:
                        if label in sizes:
                            sizes[label] += len(explored)
                        else:
                            sizes[label] = len(explored)
    logging.debug("watershed(.., T_l=%s, ..): %d new labels found"
       % (T_l, n_labels - n_labels_0))
    return n_labels


@timeit
@cython.boundscheck(False)
@cython.wraparound(False)
def get_region_graph(
        WEIGHT_DTYPE_t[:,:,:,:] aff,
        LABELS_DTYPE_t[:,:,:] labels,
        LABELS_DTYPE_t n_labels):
    """
    Create a list of edges connecting the segments in labels.

    @param aff:
        z*y*x*6 edge weight array, containing inter-pixel edge affinities
    @param labels:
        z*y*x label array, for each pixel
    @param n_labels:
        number of segments already assigned.  new segments are assigned starting with
        labels of n_labels+1.
    @return:
        Region graph of edges between neighboring regions (affinity, label1, label2)
        in decreasing order of affinities.
    """
    assert n_labels >= 3

    cdef unsigned int zsize = aff.shape[0]
    cdef unsigned int ysize = aff.shape[1]
    cdef unsigned int xsize = aff.shape[2]
    cdef unsigned int z, y, x, z1, y1, x1, i
    cdef WEIGHT_DTYPE_t f0, f1
    cdef LABELS_DTYPE_t s0, s1
    cdef tuple ss

    # Adjacency graph between neighboring segments.
    # {(s0, s1): weight}
    cdef dict segaff = {}

    # Compute the max edge weight straddling each pair of segments.
    for z in xrange(zsize):
        for y in xrange(ysize):
            for x in xrange(xsize):
                s0 = labels[z,y,x]
                if s0:
                    # Only need to check 3 directions since otherwise each edge
                    # would be checked twice.
                    for i in xrange(3):
                        f0 = aff[z,y,x,i]
                        if f0:
                            z1 = <unsigned int>(z + AFF_INDEX_MAP_c[i][0])
                            y1 = <unsigned int>(y + AFF_INDEX_MAP_c[i][1])
                            x1 = <unsigned int>(x + AFF_INDEX_MAP_c[i][2])
                            s1 = labels[z1,y1,x1]
                            if s1 and s1 != s0:
                                if s0 < s1:
                                    ss = (s0, s1)
                                else:
                                    ss = (s1, s0)
                                if ss in segaff:
                                    f1 = segaff[ss]
                                    segaff[ss] = aff_max(f0, f1)
                                else:
                                    segaff[ss] = f0

    # Create the sorted (descending) list of edge weights
    #TODO: optimize region_graph
    cdef list region_graph = sorted(
        ((f0, s0, s1) for (s0, s1), f0 in segaff.iteritems()),
        reverse=True,
        key=itemgetter(1))

    logging.debug("region_graph(...) completed")
    return region_graph


# (7f)**4
cdef inline unsigned int func_avg(WEIGHT_DTYPE_t f): return 25 if f < 0.5 else <unsigned int>(2041*f*f*f*f)
# (4f)**5
cdef inline unsigned int func_bup(WEIGHT_DTYPE_t f): return 25 if f < 0.5 else <unsigned int>(1024*f*f*f*f*f)


@timeit
@cython.boundscheck(False)
@cython.wraparound(False)
def merge_segments(
        WEIGHT_DTYPE_t[:,:,:,:] aff,
        list region_graph,
        LABELS_DTYPE_t[:,:,:] labels,
        LABELS_DTYPE_t n_labels,
        dict sizes,
        unsigned int T_s):
    """
    Merge all neighboring pairs of segments obeying func.

    @param aff:
        not really needed, temp until i can figure out how to get fused types for `f`
        working right.
    @param region_graph:
        Region graph of edges between neighboring regions (affinity, label1, label2)
        in decreasing order of affinities.
    @param labels:
        z*y*x label array, for each pixel
    @param n_labels:
        number of segments already assigned.  new segments are assigned starting with
        labels of n_labels+1.
    @param sizes:
        dict of segment sizes for each segment
    @param func:
        Function accepting an edge weight, and returning Tf_s, the minimum size
        of one segment for any pair to be merged.
    @param T_s:
        Minimum size of a segment.  After merging, all segments with size < T_s
        are discarded.
    @return:
        n_labels after this step.
        (note) labels are updated.
        (note) region_graph, sizes are NOT updated.  This was deemed unimportant
            as merge_segments isn't expected to be called recursively.
    """
    cdef LABELS_DTYPE_t n_labels_0 = n_labels
    cdef LABELS_DTYPE_t n_labels_new = 0
    cdef unsigned int zsize = labels.shape[0]
    cdef unsigned int ysize = labels.shape[1]
    cdef unsigned int xsize = labels.shape[2]
    cdef unsigned int z, y, x
    cdef unsigned int Tf_s
    cdef WEIGHT_DTYPE_t f
    cdef LABELS_DTYPE_t s0, s1, s0_root, s1_root, s2_root

    # Merge pairs of segments with 1 segment having size < Tf_s.
    #TODO: optimize uf with C++ data structures?
    cdef object uf = UnionFind()
    uf.insert_objects(xrange(1, n_labels+1))
    for f, s0, s1 in region_graph:
        s0_root = uf.find(s0)
        s1_root = uf.find(s1)
        if s0_root != s1_root:
            Tf_s = func_bup(f)
            if sizes[s0_root] < Tf_s or sizes[s1_root] < Tf_s:
                s2_root = uf.union(s0_root, s1_root)
                sizes[s2_root] = sizes[s0_root] + sizes[s1_root]
                n_labels -= 1

    # Sanity check that n_labels has been decremented properly
    if n_labels != len(uf):
        logging.error("n_labels = %d, len(uf) = %d", n_labels, len(uf))

    # Map the label roots of each pixel to (1..n_labels), discarding all
    # segments of size < T_s.
    #TODO: optimize roots, roots2labels with C++ data structures?
    cdef list roots = uf.get_roots()
    cdef dict roots2labels = {}
    for root in roots:
        if sizes[root] < T_s:
            roots2labels[root] = 0
        else:
            n_labels_new += 1
            roots2labels[root] = n_labels_new

    # Update `labels` with new labels of each pixel
    for z in xrange(zsize):
        for y in xrange(ysize):
            for x in xrange(xsize):
                if labels[z,y,x]:
                    labels[z,y,x] = roots2labels[uf.find(labels[z,y,x])]

    logging.debug("merge_segments(.., T_s=%s, ..): %d merges made, %d segments discarded"
        % (T_s, n_labels_0 - n_labels, n_labels - n_labels_new))
    return n_labels_new


@timeit
@cython.boundscheck(False)
@cython.wraparound(False)
def test_undersegmentation(
        LABELS_DTYPE_t[:,:,:] labels,
        LABELS_DTYPE_t[:,:,:] truth,
        LABELS_DTYPE_t n_labels = 0,
        LABELS_DTYPE_t n_truth = 0):

    if not n_labels:
        n_labels = np.max(labels)
    if not n_truth:
        n_truth = np.max(truth)

    cdef unsigned int zsize = labels.shape[0]
    cdef unsigned int ysize = labels.shape[1]
    cdef unsigned int xsize = labels.shape[2]
    # Sanity check
    cdef unsigned int zsize1 = truth.shape[0]
    cdef unsigned int ysize1 = truth.shape[1]
    cdef unsigned int xsize1 = truth.shape[2]
    assert zsize1 >= zsize
    assert ysize1 >= ysize
    assert xsize1 >= xsize

    cdef unsigned int z, y, x, z1, y1, x1, i, j, sz
    cdef LABELS_DTYPE_t s0

    cdef queue[unsigned int] qz, qy, qx
    cdef unsigned int mismatches = 0, tot_sz = 0
    explored_labels = set()

    for z in xrange(zsize):
        for y in xrange(ysize):
            for x in xrange(xsize):
                s0 = labels[z,y,x]
                if s0:
                    if s0 in explored_labels:
                        continue
                    qz.push(z)
                    qy.push(y)
                    qx.push(x)
                    explored = set([(z,y,x)])
                    while not qz.empty():
                        z = qz.front()
                        y = qy.front()
                        x = qx.front()
                        qz.pop()
                        qy.pop()
                        qx.pop()
                        for i in xrange(6):
                            # Check if unlabeled
                            z1 = <unsigned int>(z + AFF_INDEX_MAP_c[i][0])
                            y1 = <unsigned int>(y + AFF_INDEX_MAP_c[i][1])
                            x1 = <unsigned int>(x + AFF_INDEX_MAP_c[i][2])
                            if (z1,y1,x1) in explored:
                                continue
                            if labels[z1,y1,x1] == s0:
                                explored.add((z1,y1,x1))
                                qz.push(z1)
                                qy.push(y1)
                                qx.push(x1)
                    explored_labels = list(explored_labels)
                    sz = len(explored_labels)
                    tot_sz += <unsigned int>((sz-1)*(sz-2) / 2)
                    for i in xrange(sz):
                        for j in xrange(i, sz):
                            z,y,x = explored_labels[i]
                            z1,y1,x1 = explored_labels[j]
                            if truth[z,y,x] != truth[z1,y1,x1]:
                                mismatches += 1
    cdef result = mismatches / <float>tot_sz
    return result
