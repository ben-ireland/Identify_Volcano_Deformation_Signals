#!/usr/bin/env python

# ==========
# Overview
# ==========


# Script to identify and analyse different deformation patterns from a LiCSBAS datacube
# Ben Ireland, COMET, University of Bristol, June 2026
#
# Input: Filepath to LiCSBAS datacube to analyse

# ==== IMPORT ===
from cmcrameri import cm
from skimage import filters
from skimage.feature import peak_local_max
from skimage import measure
from skimage import morphology
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks, peak_prominences, convolve2d
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from scipy.spatial.distance import squareform
import matplotlib.colorizer as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import h5py
import numpy as np