import numpy as np
import unittest
from unittest import TestCase
from structs import formats


class TestAff(TestCase):

    def setUp(self):
        aff_3d = np.zeros((1,4,4,6), dtype=np.uint8)
        # Only need to instantiate first 3 edges, since the rest is discarded during saving
        # vertical edges
        aff_3d[0,:-1,:,1] = 10 - np.array([
            [2, 5, 8, 1],
            [3, 4, 5, 2],
            [3, 4, 7, 0],
        ])
        # horizontal edges
        aff_3d[0,:,:-1,2] = 10 - np.array([
            [1, 5, 5],
            [4, 4, 1],
            [6, 5, 0],
            [3, 4, 0],
        ])
        formats.refresh_aff(aff_3d)
        self.aff_3d = aff_3d

    def test_affv(self):
        affv = formats.aff2affv(self.aff_3d)
        affv2 = np.array([
            [9, 9, 5, 9],
            [8, 6, 9, 9],
            [7, 6, 10, 10],
            [7, 7, 10, 10],
        ])
        self.assertTrue( np.all(affv2 == affv) )


if __name__ == "__main__":
    unittest.main()
