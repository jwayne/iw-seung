"""
Implements:

Cousty et al. 2009 - Watershed Cuts: Minimum Spanning Forests and the Drop of
Water Principle
http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4564470
"""
from jpyutils.timeit import timeit
import logging
import numpy as np
import formats


@timeit
def connected_components(aff, affv, labels, threshold, n_labels):
    n_labels_0 = n_labels
    zsize, ysize, xsize, _ = aff.shape
    for z0 in xrange(zsize):
        for y0 in xrange(ysize):
            for x0 in xrange(xsize):
                if not labels[z0,y0,x0] and affv[z0,y0,x0] >= threshold:
                    n_labels += 1
                    labels[z0,y0,x0] = n_labels
                    q = [(z0,y0,x0)]
                    while q:
                        z, y, x = q.pop()
                        for i in xrange(6):
                            if aff[z,y,x,i] >= threshold:
                                # Check if unlabeled
                                z1 = z + formats.AFF_INDEX_MAP[i][0]
                                y1 = y + formats.AFF_INDEX_MAP[i][1]
                                x1 = x + formats.AFF_INDEX_MAP[i][2]
                                if not labels[z1,y1,x1]:
                                    # Note: labels also functions as explored
                                    labels[z1,y1,x1] = n_labels
                                    q.append((z1,y1,x1))
    logging.debug("connected_components:(.., threshold=%s, ..): %d new labels found"
        % (threshold, n_labels_0 - n_labels))
    return n_labels


@timeit
def watershed(aff, affv, labels, threshold, n_labels):
    n_labels_0 = n_labels
    zsize, ysize, xsize, _ = aff.shape
    for z0 in xrange(zsize):
        for y0 in xrange(ysize):
            for x0 in xrange(xsize):
                if not labels[z0,y0,x0] and affv[z0,y0,x0] >= threshold:
                    q = [(z0,y0,x0)]
                    explored = set([(z0,y0,x0)])
                    label = 0
                    while q:
                        z, y, x = q.pop()
                        for i in xrange(6):
                            if aff[z,y,x,i] >= threshold and aff[z,y,x,i] == affv[z,y,x]:
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
    logging.debug("watershed(.., threshold=%s, ..): %d new labels found"
        % (threshold, n_labels_0 - n_labels))
    return n_labels
