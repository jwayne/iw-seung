"""
Implements:

Cousty et al. 2009 - Watershed Cuts: Minimum Spanning Forests and the Drop of
Water Principle
http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4564470
"""
from jpyutils.timeit import timeit
from jpyutils.unionfind import UnionFind
import logging
import numpy as np
import formats


@timeit
def connected_components(aff, affv, T_h, labels, n_labels=0, sizes=None):
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
    n_labels_0 = n_labels
    zsize, ysize, xsize, _ = aff.shape
    for z0 in xrange(zsize):
        for y0 in xrange(ysize):
            for x0 in xrange(xsize):
                if not labels[z0,y0,x0] and affv[z0,y0,x0] >= T_h:
                    n_labels += 1
                    labels[z0,y0,x0] = n_labels
                    if sizes is not None:
                        sizes[n_labels] += 1
                    q = [(z0,y0,x0)]
                    while q:
                        z, y, x = q.pop()
                        # Only need to check 3 directions since otherwise each edge
                        # would be checked twice.
                        for i in xrange(3):
                            if aff[z,y,x,i] >= T_h:
                                # Check if unlabeled
                                z1 = z + formats.AFF_INDEX_MAP[i][0]
                                y1 = y + formats.AFF_INDEX_MAP[i][1]
                                x1 = x + formats.AFF_INDEX_MAP[i][2]
                                if not labels[z1,y1,x1]:
                                    # Note: labels also functions as explored
                                    labels[z1,y1,x1] = n_labels
                                    if sizes:
                                        sizes[n_labels] += 1
                                    q.append((z1,y1,x1))
    logging.debug("connected_components:(.., T_h=%s, ..): %d new labels found"
        % (T_h, n_labels - n_labels_0))
    return n_labels


@timeit
def watershed(aff, affv, T_l, labels, n_labels=0, sizes=None):
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
    n_labels_0 = n_labels
    zsize, ysize, xsize, _ = aff.shape
    for z0 in xrange(zsize):
        for y0 in xrange(ysize):
            for x0 in xrange(xsize):
                if not labels[z0,y0,x0] and affv[z0,y0,x0] >= T_l:
                    q = [(z0,y0,x0)]
                    explored = set([(z0,y0,x0)])
                    label = 0
                    while q:
                        z, y, x = q.pop()
                        # Need to check 6 directions since otherwise certain 'under'
                        # directions would be missed.
                        for i in xrange(6):
                            if aff[z,y,x,i] >= T_l and aff[z,y,x,i] == affv[z,y,x]:
                                # Check if unlabeled
                                z1 = z + formats.AFF_INDEX_MAP[i][0]
                                y1 = y + formats.AFF_INDEX_MAP[i][1]
                                x1 = x + formats.AFF_INDEX_MAP[i][2]
                                if (z1,y1,x1) in explored:
                                    continue
                                if labels[z1,y1,x1]:
                                    # Found inf-stream under this stream, so 
                                    label = labels[z1,y1,x1]
                                    q = None
                                    break
                                elif affv[z1,y1,x1] > affv[z,y,x]:
                                    # Found new bottom for this stream, so replace q
                                    # TODO: explore if taking max of all matches here affects result?
                                    explored.add((z1,y1,x1))
                                    q = [(z1,y1,x1)]
                                    break
                                elif affv[z1,y1,x1] == affv[z,y,x]:
                                    # Found equivalent possible bottoms for this stream, so augment q
                                    explored.add((z1,y1,x1))
                                    q.append((z1,y1,x1))
                                else:
                                    assert False
                    if not label:
                        n_labels += 1
                        label = n_labels
                    for z2,y2,x2 in explored:
                        labels[z2,y2,x2] = label
                    if sizes:
                        sizes[label] += len(explored)
    logging.debug("watershed(.., T_l=%s, ..): %d new labels found"
       % (T_l, n_labels - n_labels_0))
    return n_labels


@timeit
def get_region_graph(aff, labels, n_labels):
    """
    Create a list of edges connecting the segments in labels.
    @return:
        Region graph of (affinity, label1, label2)
        in decreasing order of affinities.
    """
    # compact format of the adjacency graph, where the weight between segments
    # s0,s1 is region_graph[(s0-2)*(s0-1)/2 + (s1-1)]
    N = (n_labels-2)*(n_labels+1)/2
    aff_segments = np.zeros(N)
    zsize, ysize, xsize, _ = aff.shape

    # Compute the max edge weight straddling each pair of segments.
    for z0 in xrange(zsize):
        for y0 in xrange(ysize):
            for x0 in xrange(xsize):
                s0 = labels[z0,y0,x0]
                if s0:
                    # Only need to check 3 directions since otherwise each edge
                    # would be checked twice.
                    for i in xrange(3):
                        f = aff[z0,y0,x0,i]
                        if f:
                            z1 = z0 + formats.AFF_INDEX_MAP[i][0]
                            y1 = y0 + formats.AFF_INDEX_MAP[i][1]
                            x1 = x0 + formats.AFF_INDEX_MAP[i][2]
                            s1 = labels[z1,y1,x1]
                            if s1 and s1 != s0:
                                if s0 < s1:
                                    # Swap
                                    s0 += s1
                                    s1 = s0 - s1
                                    s0 = s0 - s1
                                ind = (s0-2)*(s0-1)/2+s1-1
                                aff_segments[ind] = max(aff_segments[ind], f)

    # Create the sorted (descending) list of edge weights
    region_graph = []
    ind = 0
    for s0 in xrange(2, n_labels+1):
        for s1 in xrange(1, s0):
            region_graph.append((aff_segments[ind], s0, s1))
    region_graph = sorted(region_graph, reverse=True)

    logging.debug("region_graph(...) completed")
    return region_graph


@timeit
def merge_segments(region_graph, labels, n_labels, T_e, T_s, sizes):
    """
    @return:
        n_labels after this step.
        (note) labels are updated.
        (note) region_graph, sizes are NOT updated.  This was deemed
        unimportant as merge_segments isn't expected to be called recursively.
    """
    n_labels_0 = n_labels

    uf = UnionFind()
    uf.insert_objects(xrange(1, n_labels+1))
    for f, s0, s1 in region_graph:
        ss0 = uf.find(s0)
        ss1 = uf.find(s1)
        if ss0 != ss1 and f >= T_e:
            if sizes[ss0] <= T_s or sizes[ss1] <= T_s:
                ssX = uf.union(ss0, ss1)
                sizes[ssX] = sizes[ss0] + sizes[ss1]
                n_labels -= 1

    # TODO: Sanity check that n_labels is right?
    n_labels = len(uf)
    # Map from old labels to new
    label_map = dict((root,i+1) for i,root in enumerate(uf.get_roots()))

    # Discard all segments of size < T_s.
    sizes2 = dict((new_label, sizes[root]) for root,new_label in label_map.iteritems())
    for label, size in sizes2.iteritems():
        if size < T_s:
            label_map[label] = 0
    # TODO: update sizes (return?)

    # Update `labels` with new labels of each pixel
    zsize, ysize, xsize = labels.shape
    for z0 in xrange(zsize):
        for y0 in xrange(ysize):
            for x0 in xrange(xsize):
                labels[z0,y0,x0] = label_map[uf.find(labels[z0,y0,x0])]

    # TODO: update region_graph (return?)

    logging.debug("merge_segments(.., T_e=%s, T_s=%s, ..): %d merges made"
        % (T_e, T_s, n_labels_0 - n_labels))
    return n_labels
