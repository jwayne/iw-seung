"""
Evaluate labeling success.
"""
import sys
import config
from structs import formats
from oversegmenters.watershed_util import test_undersegmentation

labels = formats.read_labels(sys.argv[1])
truth = formats.read_labels(config.fn_truth)
rate = test_undersegmentation(labels, truth)

print rate
