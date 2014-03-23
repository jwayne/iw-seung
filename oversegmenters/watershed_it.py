"""
Iteratively threshold the affinity graph at X and grow objects to an affinity
value of Y < X, for decreasing pairs (X,Y).

Then, perform a "distance transform based object-breaking watershed procedure"
to "slightly reduce the rate of undersegmentation in large objects".

Finish off by growing all objects to an affinity value of 0.2.

Bogovic, Huang, Jain 2013 - Learned vs Hand-Designed Features for Agglo
Cousty 2009
"""
import logging
import numpy as np
from jpyutils.timeit import timeit

import formats



@timeit
def connected_components(aff, affv, labels, threshold, n_labels):
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
    return n_labels


@timeit
def watershed(aff, affv, labels, threshold, n_labels):
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
    return n_labels



def oversegment_aff(aff_3d):
    zsize, ysize, xsize, nedges = aff_3d.shape
    assert nedges == 6
    # Normally, we won't be exploring out-of-bounds vertices because
    # edges going out-of-bounds are always 0, but if '--lim' is
    # used, then those edges might be nonzero.
    # Thus, we should check limits :(
    aff_3d[-1, :, :, 0] = 0
    aff_3d[:, -1, :, 1] = 0
    aff_3d[:, :, -1, 2] = 0
    aff_3d[0, :, :, 3] = 0
    aff_3d[:, 0, :, 4] = 0
    aff_3d[:, :, 0, 5] = 0

    # Set each vertex's weight to the max of its adjacent edges
    labels_3d = np.zeros(aff_3d.shape[:-1], dtype=formats.LABELS_DTYPE)
    affv_3d = formats.aff2affv(aff_3d)

    n_labels = 0
#    n_labels = watershed(aff_3d, affv_3d, labels_3d, 1, n_labels)
    for t_cc, t_ws in ((.9,.8), (.8,.7), (.7,.6), (.6,.2)):
        t_cc *= formats.AFF_MAX
        t_ws *= formats.AFF_MAX
        n_labels = connected_components(aff_3d, affv_3d, labels_3d, t_cc, n_labels)
        print n_labels
        # We are allowed to do this because all minima are labeled due to
        # the previous function.
        n_labels = watershed(aff_3d, affv_3d, labels_3d, t_ws, n_labels)
        print n_labels


    return labels_3d, n_labels
