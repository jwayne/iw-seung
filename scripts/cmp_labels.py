"""
Compare 2 label outputs and see if they're the same.
"""
from structs.formats import read_labels
import numpy as np
import sys

x=read_labels(sys.argv[1])
y=read_labels(sys.argv[2])
compared = (x==y)
print compared
print np.all(compared)
