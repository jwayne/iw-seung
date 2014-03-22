"""
Take a tiff file containing oversegmented regions, and produce/save pairs of
neighboring regions with internal pixels containing their original boundary
map values, and external pixels masked out.
"""
import argparse
import Image
import numpy as np
from scipy import stats
import os
import random


class NeighborSearch(object):
    """
    Build a reasonably efficient filter that can search a fixed radius
    around pixels of interest, for pixels matching some criterion.
    """

    def __init__(self, radius):
        radius2 = radius*radius
        self.diffs = np.array([(dx,dy) for dx in xrange(-radius,radius+1)
                          for dy in xrange(-radius,radius+1)
                          if (1 < dx*dx+dy*dy <= radius2)])
        self.test_diffs = np.array([(1,0),(0,1),(0,-1),(-1,0)])


    def find_neighbor_labels(self, arr_labels, arr_binary_i):
        """
        Find labels of pixels in arr_labels, neighboring the group defined
        by arr_binary_i (which is a binary image with True's at pixels belonging
        to the label of interest).
        """
        inds = arr_binary_i.nonzero()
        inds = np.transpose(inds)
        label = arr_labels[tuple(inds[0])]

        neighbor_labels = set()

        for ind in inds:
            neighbor_inds = self.test_diffs + ind
            neighbor_inds.clip(0, 1023, neighbor_inds)
            neighbor_inds = tuple(neighbor_inds.transpose())
            this_neighbor_labels = arr_labels[neighbor_inds]

            # Skip if this pixel is surrounded by pixels of the same label
            if np.all(this_neighbor_labels == label):
                continue

            neighbor_labels.update(this_neighbor_labels)

            neighbor_inds = self.diffs + ind
            neighbor_inds.clip(0, 1023, neighbor_inds)
            neighbor_inds = tuple(neighbor_inds.transpose())
            neighbor_labels.update(arr_labels[neighbor_inds])

        if label in neighbor_labels:
            neighbor_labels.remove(label)

        return neighbor_labels


def prepare_agglomeration_pairs(arr_gs, arr_labels, arr_gt, n_labels, dir_data):
    """
    For pairs of groups in arr_labels, prepare:

      (1) images consisting of the gs values in the OR of pixels in those pairs
          and black for all other pixels

      (2) a score for the pair approximating the likelihood that the
          pair belongs to the same group in the final segmentation.

    (1) and (2) can be taken as the input and output, respectively, of a
    classifier predicting
    the likelihood of two pairs belonging to the same final group.
    """
    binary_labels = {}
    def get_arr_binary(lab):
        if lab in binary_labels:
            arr_binary = binary_labels[lab]
        else:
            arr_binary = (arr_labels == lab)
            binary_labels[lab] = arr_binary
        return arr_binary

    RADIUS = 4
    FRAC_TRAIN = .75
    ns = NeighborSearch(RADIUS)


    dir_train = os.path.join(dir_data, 'agglo_train')
    dir_test = os.path.join(dir_data, 'agglo_test')
    if os.path.exists(dir_train):
        shutil.rmtree(dir_train)
    os.mkdir(dir_train)
    if os.path.exists(dir_test):
        shutil.rmtree(dir_test)
    os.mkdir(dir_test)

    print "Num labels: %d" % n_labels
    count = 0

    explored = set()
    for i in xrange(1, n_labels+1):
        arr_binary_i = get_arr_binary(i)
        # Find neighbors of group i
        neighbor_labels = ns.find_neighbor_labels(arr_labels, arr_binary_i)

        for j in neighbor_labels:
            if (i,j) in explored:
                continue
            explored.add((i,j))

            arr_binary_j = get_arr_binary(j)
            # Make masked image of pair (input)
            arr_gs_pair = np.copy(arr_gs)
            arr_mask_pair = np.invert(np.logical_or(arr_binary_i, arr_binary_j))
            arr_gs_pair[arr_mask_pair] = 0

            # Compute likelihood (output)
            # This is really simple for now, just checks if the modes of the two
            # are the same...
            mode_i = stats.mode(arr_gt[arr_binary_i])[0][0]
            mode_j = stats.mode(arr_gt[arr_binary_j])[0][0]
            in_same_group = (mode_i == mode_j)

            if random.random() < FRAC_TRAIN:
                dir_out = dir_train
            else:
                dir_out = dir_test

            im_pair = Image.fromarray(np.uint8(arr_gs_pair))
            im_pair.save(os.path.join(dir_out, '%09d-%d.tif' % (count, in_same_group)))

            count += 1
        print "%d/%d labels, %d images so far" % (i, n_labels, count)

    return



def main(dir_data):

    # Open stack of slices stored as tiff
    tiff_gs = Image.open(os.path.join(dir_data, 'train-input.tif'))
    tiff_bm = Image.open(os.path.join(dir_data, 'train-boundarymap.tif'))
    tiff_gt = Image.open(os.path.join(dir_data, 'train-labels.tif'))
    i = 0
    while True:
        try:
            tiff_gs.seek(i)
            tiff_bm.seek(i)
            tiff_gt.seek(i)
        except EOFError:
            break
        i += 1

        arr_gs = np.array(tiff_gs.getdata()).reshape(tiff_gs.size)
        arr_bm = np.array(tiff_bm.getdata()).reshape(tiff_bm.size)
        arr_gt = np.array(tiff_gt.getdata()).reshape(tiff_gt.size)

        arr_labels, n_labels = run_watershed(arr_bm)

        prepare_agglomeration_pairs(arr_gs, arr_labels, arr_gt, n_labels, dir_data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_data')
    args = parser.parse_args()

    main(args.dir_data)
