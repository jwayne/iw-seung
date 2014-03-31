"""
Iteratively threshold (connected components) the affinity graph at X,
and grow objects (watershed) to an affinity value of Y < X, for decreasing
pairs (X,Y).

Then, perform a "distance transform based object-breaking watershed procedure"
to "slightly reduce the rate of undersegmentation in large objects".

Finish off by growing all objects to an affinity value of 0.2.

Bogovic, Huang, Jain 2013 - Learned vs Hand-Designed Feature Representations
for 3d Agglomeration
"""
from jpyutils.timeit import timeit
import numpy as np
from structs import formats
from oversegmenters.watershed_util import connected_components, watershed


@timeit
def oversegment_aff(aff_3d):
    # Set each vertex's weight to the max of its adjacent edges
    affv_3d = formats.aff2affv(aff_3d)

    labels_3d = np.zeros(aff_3d.shape[:-1], dtype=formats.LABELS_DTYPE)
    n_labels = 0
    aff_dtype = aff_3d.dtype

    for t_cc, t_ws in ((.9,.8),):# (.8,.7), (.7,.6), (.6,.2)):
        if aff_dtype == formats.WEIGHT_DTYPE_UINT:
            t_cc = int(t_cc * formats.WEIGHT_MAX_UINT)
            t_ws = int(t_ws * formats.WEIGHT_MAX_UINT)

        n_labels = connected_components(aff_3d, affv_3d, t_cc, labels_3d, n_labels)

        # We are allowed to apply watershed with some pixels already labeled
        # and without a quick-union structure because, due to the
        # connected_components algo, those labels always contain mins.
        n_labels = watershed(aff_3d, affv_3d, t_ws, labels_3d, n_labels)

    return labels_3d, n_labels
