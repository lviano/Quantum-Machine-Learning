import numpy as np
import matplotlib.pyplot as plt

import user_interface as ui
import visualization as viz
import debug
import sys

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from validation import *
from load_data import*
from sklearn.preprocessing import scale

def pca_projection(x, rank):
    x_pca = PCA(n_components = int(rank)).fit_transform(x)

    return x_pca

def cluster(names, x, args):
    # Estimate the rank
    rank = estimate_rank(x, method = args.rank_estimation, plot = args.plot)

    # PCA projection onto the space with dimension of the estimated rank n
    x_pca_n = pca_projection(x, rank)

    # DBSCAN clustering according to the new PCA projection space
    db = DBSCAN(eps = args.epsilon, min_samples = args.min_samples).fit(x_pca_n) # eps and min_samples are estimated by a k-distance plot

    # 2-dimensional PCA-projection for visualization
    x_pca_2d = pca_projection(x_pca_n, 2)

    return rank, x_pca_n, x_pca_2d, db

def main(argv):
    # Read command line options
    args = ui.read_command_line_options(argv)

    # Initialize logging
    debug.init_log(sys.stderr, args.logging)

    # Load features
    names, x = load_features(args.path, args.file, args.normalize, args.add_information, args.expand, args.degree)
    #x_copy = np.copy(x)

    # Apply PCA projection and DBSCAN clustering
    rank, x_pca_n, x_pca_2d, db = cluster(names, x, args)

    #Plot Cluster if specified
    if args.plot:
        viz.visualize(args, names, rank, db, x_pca_2d)

if __name__ == '__main__':
    main(sys.argv)

