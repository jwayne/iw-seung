"""
Take 2D slices one at a time, and segment each using standard watershed
with local mins as seeds.
"""
import logging
import numpy as np
from scipy import ndimage
from skimage.morphology import watershed
from skimage.feature import peak_local_max



def _oversegment_bm_slice(bm_2d):
    # Find local max's to use as markers for where to start watershed
    localmax_2d = peak_local_max(bm_2d, indices=False)
    markers_2d, n_markers = ndimage.label(localmax_2d)

    # Threshold boundary pixels to be left unlabeled?
    #mask_2d = (bm_2d < FRAC)

    # Run watershed
    labels_2d = watershed(-bm_2d, markers_2d)
    #labels_2d = watershed(-bm_2d, markers_2d, mask=mask_2d)

    # Shuffle since label identities are technically arbitrary.
#    inds = np.arange(1,n_markers+1)
#    np.random.shuffle(inds)
#    for old_ind, new_ind in enumerate(inds):
#        old_ind += 1
#        labels_2d[labels_2d == old_ind] = new_ind

    return labels_2d, n_markers


def oversegment_bm(bm_3d):
    labels_3d = np.zeros(bm_3d.shape)
    tot_labels = 0
    for i, bm_2d in enumerate(bm_3d):
        labels_2d, n_labels = _oversegment_bm_slice(bm_2d)
        # Adjust label values since regions technically don't share labels across slices.
#        labels_2d += tot_labels
        labels_3d[i,:,:] = labels_2d
        tot_labels += n_labels
        logging.info("Oversegmented slice %d/%d, %d labels" % (i+1, len(bm_3d), n_labels))
    return labels_3d, tot_labels
