"""
Take 2D slices one at a time, and segment each using standard watershed
with local mins as seeds.
"""
import logging
import numpy as np
from scipy import ndimage
from skimage.morphology import watershed
from skimage.feature import peak_local_max



def _oversegment_slice(slice_bm):
    # Find local max's to use as markers for where to start watershed
    slice_localmax = peak_local_max(slice_bm, indices=False)
    slice_markers, n_markers = ndimage.label(slice_localmax)

    # Threshold boundary pixels to be left unlabeled?
    #slice_mask = (slice_bm < FRAC)

    # Run watershed
    slice_labels = watershed(-slice_bm, slice_markers)
    #slice_labels = watershed(-slice_bm, slice_markers, mask=slice_mask)

    # Shuffle since label identities are technically arbitrary.
#    inds = np.arange(1,n_markers+1)
#    np.random.shuffle(inds)
#    for old_ind, new_ind in enumerate(inds):
#        old_ind += 1
#        slice_labels[slice_labels == old_ind] = new_ind

    return slice_labels, n_markers


def oversegment_bm(stack_bm):
    stack_labels = np.zeros(stack_bm.shape)
    tot_labels = 0
    for i, slice_bm in enumerate(stack_bm):
        slice_labels, n_labels = _oversegment_slice(slice_bm)
        # Adjust label values since regions technically don't share labels across slices.
#        slice_labels += tot_labels
        stack_labels[i,:,:] = slice_labels
        tot_labels += n_labels
        logging.info("Oversegmented slice %d/%d, %d labels" % (i+1, len(stack_bm), n_labels))
    return stack_labels, tot_labels
