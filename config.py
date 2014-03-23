"""
Specifies which data files to use.

dn = directory name
fn = file name
"""
import os

dn_data = "../data/"
fn_bm = "train-membranes-idsia.tif"
fn_aff = "toy_aff.raw"
shape_aff = (1, 4, 4)
#fn_aff = "train-membranes-idsia.raw"
#shape_aff = (100, 1024, 1024)

dn_data = os.path.abspath(dn_data)
fn_bm = os.path.join(dn_data, fn_bm)
fn_aff = os.path.join(dn_data, fn_aff)
