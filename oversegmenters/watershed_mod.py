"""
Threshold (connected components) the affinity graph at T_h, the 'high
threshold', and grow objects (watershed) to an affinity value of T_l,
the 'low threshold'.

Then, take regions and their edges in decreasing order of affinity

Aleksandar Zlateski 2011 - A design and implementation of an efficient,
parallel watershed algorithm for affinity graphs (master's thesis)
http://dspace.mit.edu/handle/1721.1/66820
"""
from collections import defaultdict
from jpyutils.timeit import timeit
import numpy as np
from structs import formats
from oversegmenters.watershed_util import connected_components, watershed, get_region_graph, merge_segments

#TODO: parallelizing? how does my discretizing of the affinity graph (i.e., the
# enabling of many non-maxima plateaus) prevent parallelization? (see 4.1?)


@timeit
def oversegment_aff(aff_3d):
    # Thresholds from Zletaski 2009, p. 19
    """
    T_h = 0.98
    T_l = 0.2
    T_e = 0.1
    T_s = 25
    """

    # Thresholds from Aleks's code.
    T_h = 0.99
    T_l = 0.3
    T_e = 0.1 #not used
    T_s = 25

    if aff_3d.dtype == formats.WEIGHT_DTYPE_UINT:
        T_h = int(T_h * formats.WEIGHT_MAX_UINT)
        T_l = int(T_l * formats.WEIGHT_MAX_UINT)
        T_e = int(T_e * formats.WEIGHT_MAX_UINT)

    # Set each vertex's weight to the max of its adjacent edges
    affv_3d = formats.aff2affv(aff_3d)

    labels_3d = np.zeros(aff_3d.shape[:-1], dtype=formats.LABELS_DTYPE)
    n_labels = 0
    sizes = {}

    # Watershed with edges >= T_h merged, and edges < T_l not considered
    n_labels = connected_components(aff_3d, affv_3d, T_h, labels_3d, n_labels, sizes)
    # We are allowed to apply watershed with some pixels already labeled
    # and without a quick-union structure because, due to the
    # connected_components algo, those labels always contain mins.
    n_labels = watershed(aff_3d, affv_3d, T_l, labels_3d, n_labels, sizes)

    # Create the region graph, and list their edges in decreasing order
    # Ignore unlabeled vertices, since they are "single-vertex segments" and
    # I think they'll be unlikely to reach size > T_s even after the next step.
#    region_graph = get_region_graph(aff_3d, labels_3d, n_labels)

    # For all edges with affinity >= T_e, merge if any segment has size < T_s.
#    n_labels = merge_segments(aff_3d, region_graph, labels_3d, n_labels, sizes, T_s)

    return labels_3d, n_labels
