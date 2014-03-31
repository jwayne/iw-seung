iw-seung
========

Segment a 3D boundary map or affinity graph using various segmentation algorithms.

Code accelerated via Cython, and is 1.5-2x faster than equivalent C++ code (see [https://github.com/jwayne/aleks-watershed]) with several easy optimizations still unimplemented.

My goal is to use this code to rapidly develop and test different agglomeration algorithms.


# Getting Started

Get the dependencies:
* jpyutils [https://github.com/jwayne/jpyutils]
* tifffile (`sudo easy_install tifffile`)
* numpy
* g++ (to compile C++ files generated by Cython)
* optional-- Cython v0.20+ (to compile .pyx files from scratch)
* optional-- scipy (some outdated files)
* optional-- scikit-image (some outdated files)

Compile the Cython extensions.  (Note that Cython is not needed The following command is assuming you want to use this package in its existing directory:
```
./setup.py build_ext --inplace
```

Run oversegment.py.
```
# Run `watershed_it` algorithm on a given boundary map, like that downloadable from SNEMI3D
./oversegment.py watershed_it -i boundary_map.tif -o labels.tif

# Run `watershed_it` algorithm on a given affinity graph, obtainable via scripts/bm2aff.py
./oversegment.py watershed_it -i affinity_graph.raw -o labels.tif

# Run `watershed_it` algorithm on a given affinity graph, like that used in aleks-watershed.
# There are some finicky restrictions on the filename here, though (filename must match exactly
# with aleks's benchmark affinity graph)
./oversegment.py watershed_it -i affinity_graph.raw -o labels.tif

# Help
./oversegment.py --help
```
Other useful scripts can be found in `scripts/`.
