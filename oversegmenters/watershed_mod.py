"""
Threshold (connected components) the affinity graph at T_h, the 'high
threshold', and grow objects (watershed) to an affinity value of T_l,
the 'low threshold'.

Then, take regions and their edges in decreasing order of affinity

Aleksandar Zlateski 2011 - A design and implementation of an efficient,
parallel watershed algorithm for affinity graphs (master's thesis)
http://dspace.mit.edu/handle/1721.1/66820
"""
import numpy as np
import formats
from oversegmenters.watershed_util import connected_components, watershed

#TODO: what to do about single-vertex regions? (i.e., unlabeled vertices after
# the initial watershed
#TODO: parallelizing? how does my discretizing of the affinity graph (i.e., the
# enabling of many non-maxima plateaus) prevent parallelization? (see 4.1?)


def oversegment_aff(aff_3d):
    zsize, ysize, xsize, nedges = aff_3d.shape
    assert nedges == 6

    # Thresholds from p. 19
    T_h = 0.98
#    T_h = 0.6
    T_l = 0.2
    T_e = 0.1
    T_s = 25

    # Set each vertex's weight to the max of its adjacent edges
    affv_3d = formats.aff2affv(aff_3d)

    labels_3d = np.zeros(aff_3d.shape[:-1], dtype=formats.LABELS_DTYPE)
    n_labels = 0

    t_cc *= formats.AFF_MAX
    t_ws *= formats.AFF_MAX

    # Watershed with edges >= T_h merged, and edges < T_l not considered
    n_labels = connected_components(aff_3d, affv_3d, labels_3d, T_h, n_labels)
    # We are allowed to apply watershed with some pixels already labeled
    # and without a quick-union structure because, due to the
    # connected_components algo, those labels always contain mins.
    n_labels = watershed(aff_3d, affv_3d, labels_3d, T_l, n_labels)

    # Create the region graph, and list their edges in decreasing order
    # Ignore unlabeled vertices, since they are "single-vertex segments" and
    # I think they'll be unlikely to reach size > T_s even after the next step.
    
    # For all edges with affinity >= T_e, merge if any segment has size < T_s.
    # NOTE: T_s increases with affinity in the new version of MW
    
    # Discard all segments of size < T_s.
