import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import gridspec

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from validation import *
from load_data import*
from element_information import *
from ligand_information import *

def prepare_data(path, expand, degree):
    """
    """
    names, tx = load_features(path , "proposition_of_features.txt", expand=expand, add_information = True, degree=degree)
    energies, names2 = load_energies(path + "comp_BoB_reordered_energies.txt")
    tx = np.delete(tx, np.arange(8,11), axis = 1)
    #names, tx = select_elements_from_data(names, tx, ["Pt"], remove = False)
    #names2, energies = select_energies_from_data(names2, energies, ["Pt"], remove = False)

    indices = np.random.permutation(names.shape[0])
    indices2 = np.random.permutation(names.shape[0])
    return names[indices], tx[indices], energies[indices]

def PCA_DBSCAN(tx, rank, eps, min_samples):
    """
    """
    x = PCA(n_components = rank).fit_transform(tx)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(x)

    labels = db.labels_

    x2d = PCA(n_components=rank).fit_transform(tx)
    return x2d, labels

def scatter_plot(x2d, energies, labels, names):
    """
    """
    min = np.min(energies)
    max = np.max(energies)

    ligands = get_ligands_array(names)

    unique_labels = set(labels)

    cols = 3
    rows = int(3*(math.ceil(len(unique_labels) - 1)/cols))
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure()

    print(np.unique(ligands[:,0], return_counts=True)[1])
    left, right = get_ligand_distribution(ligands)


    for i, k in enumerate(unique_labels):
        if (k != -1):
            indices = (labels == k)
            #Print number of elements
            print("Number of elements in set : " + str(names[indices].shape[0]))
                
            ax = fig.add_subplot(gs[3*i])

            ax.scatter(x2d[:,0], x2d[:,1], c=energies, cmap='hsv', alpha=0)
            clb = ax.scatter(x2d[indices,0], x2d[indices,1], c=energies[indices], cmap='hsv', s=10, alpha=0.05, vmin=min, vmax=max)
            plt.colorbar(clb, alpha=1)

            ax = fig.add_subplot(gs[3*i + 1])

            unique, counts = np.unique(ligands[:,0][indices], return_counts=True)
            ax.bar(unique, counts)
            ax.set_xlim(0, 90)

            ax = fig.add_subplot(gs[3*i + 2])

            unique, counts = np.unique(ligands[:,1][indices], return_counts=True)
            ax.bar(unique, counts)
            ax.set_xlim(0, 90)
        

        else: break

    plt.show()

rank = 3 #from estimate_rank
min_samples = 4
eps = 0.4

names, data_matrix, energies = prepare_data('../data/', False, 2)

x2d, labels = PCA_DBSCAN(data_matrix, rank, eps, min_samples)

scatter_plot(x2d, energies, labels, names)
