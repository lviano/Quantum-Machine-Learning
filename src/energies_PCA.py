import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import math

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from validation import *
from load_data import*
from element_information import *
from ligand_information import *

#Activate latex in plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
figsize = (10, 8)
fontsize = 24
colormap = 'gist_rainbow'
dpi = 300

def prepare_data(path, expand, degree):
    """
    Loads the data (molecules properties and their associated energies)

    Args:
        path (str): folder containing the two files

    """
    names, tx = load_features(path , "proposition_of_features.txt", expand=expand, degree=degree)
    _, not_normalized_data = load_features(path, "proposition_of_features.txt", expand=False, normalize = False)
    energies, _ = load_energies(path + "comp_BoB_reordered_energies.txt")

    indices = np.random.permutation(names.shape[0])
    return names[indices], tx[indices], energies[indices], not_normalized_data[indices]

def PCA_DBSCAN(tx, rank, eps, min_samples):
    """
    Applies a PCA with the desired rank and then find the clusters with DBSCAN and returns a 2D projection

    Args:
        tx (matrix): data matrix normalized of the molecules
        rank (int): desired rank for the PCA
        eps (float): eps value for DBSCAN
        min_samples (int): minimum samples per cluster for DBSCAN

    Returns:
        x2d: 2D projection of the clustered data
        labels: labels defined by DBSCAN for each point that can be used to select indiviual clusters
    """
    x = PCA(n_components = rank).fit_transform(tx)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(x)

    labels = db.labels_

    x2d = PCA(n_components=2).fit_transform(tx)
    return x2d, labels

def scatter_plot_energies(x2d, energies, savefig="", x1_sub=-2.06, x2_sub=-1.70, y1_sub=0.96, y2_sub=1.37):
    """
    Plots the 2D PCA of the data with the associated energies
    """
    #For the colorbar, find the min and max energies
    min = np.min(energies)
    max = np.max(energies)

    #===Main plot===
    #   Title of plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_title(r"2D PCA with the associated $\Delta G (\textrm{Rxn}$ $A)$ energies", fontsize=fontsize)

    sc = ax.scatter(x2d[:,0], x2d[:,1], c=energies, cmap=colormap, alpha=0.05, s=10, vmin=min, vmax=max)
    cb = plt.colorbar(sc, label=r"$\Delta G (\textrm{Rxn}$ $A)$ [kcal/mol]")
    cb.set_alpha(0.5)
    cb.draw_all()

    #===Zoomed plot===
    axin = zoomed_inset_axes(ax, 4, loc=2)
    axin.scatter(x2d[:,0], x2d[:,1], c=energies, cmap=colormap, alpha=0.1, s=10, vmin=min, vmax=max)
    #   sub region of the original image
    axin.set_xlim(x1_sub, x2_sub)
    axin.set_ylim(y1_sub, y2_sub)

    plt.xticks(visible=False)
    plt.yticks(visible=False)

    #   Draw a bow around the subplot
    mark_inset(ax, axin, loc1=1, loc2=3, fc="none", ec="0.5")

    plt.tight_layout()
    if savefig != "":
        fig.savefig(savefig, dpi=dpi)

    plt.show()

def plot_cluster_ligand_count(x2d, energies, labels, names, savefigs=""):
    """
    Plots the histogram of the ligands (left and right) of all individual clusters found by DBSCAN
    """
    #For the colorbar, find the min and max energies
    min = np.min(energies)
    max = np.max(energies)

    unique_labels = set(labels)
    ligands = get_ligands_array(names)

    _, counts_left = np.unique(ligands[:,0], return_counts=True)
    _, counts_right = np.unique(ligands[:,1], return_counts=True)

    total_ligands_nbr = (counts_left + counts_right)[0]

    for i, k in enumerate(unique_labels):
        indices = (labels == k)
        mask = np.ones(labels.shape, bool)
        mask[indices] = False

        unique_ligands = np.unique(ligands[:,0])

        #===Count of the left and right ligand species===
        unique_left, counts_left = np.unique(ligands[:,0][indices], return_counts=True)
        unique_right, counts_right = np.unique(ligands[:,1][indices], return_counts=True)

        counts = np.zeros(unique_ligands.shape)
        counts[unique_left] = counts_left
        counts[unique_right] += counts_right
        counts /= total_ligands_nbr

        #===Scatter plot===
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(211)
        ax.scatter(x2d[mask,0], x2d[mask,1], c='gray', s=10, alpha=0.01)
        ax.set_title(r"Selected cluster", fontsize=fontsize)
        sc = ax.scatter(x2d[indices,0], x2d[indices,1], c=energies[indices], cmap='hsv', s=10, alpha=0.05, vmin=min, vmax=max)
        cb = plt.colorbar(sc, label=r"$\Delta G (\textrm{Rxn}$ $A)$ [kcal/mol]")
        cb.set_alpha(0.5)
        cb.draw_all()

        plt.subplots_adjust(hspace = 0.3)

        #===Histogram
        ax2 = fig.add_subplot(212)
        ax2.set_title(r"Ligands count in selected molecules cluster", fontsize=fontsize)
        ax2.bar(unique_ligands, counts)
        ax2.set_xlabel(r'Ligand species number')
        ax2.set_ylabel(r'Percentage of all ligands')
        ax2.set_xlim(0, 90)
        fig.tight_layout()

        if savefigs != "":
            fig.savefig(savefigs + str(k) + ".png", dpi=dpi)


    plt.show()

def scatter_plot_molecular_weight(x2d, data_matrix, savefig=""):
    """
    Plots the 2D PCA with the molecules colored by their molecular weight
    """

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_title(r"2D PCA with molecules colored by their molecular volume", fontsize=fontsize)
    sc = ax.scatter(x2d[:,0], x2d[:,1], c=data_matrix[:,2], alpha=0.05, s=10, cmap=colormap)
    cb = plt.colorbar(sc, label=r"Molecular volume [m$^3$/mol]")
    cb.set_alpha(0.5)
    cb.draw_all()

    fig.tight_layout()
    if savefig != "":
        fig.savefig(savefig, dpi=dpi)

    plt.show()

def plot_combination_colors(x2d, data_matrix, savefig=""):
    """
    Tries to color code the last four features in the data set and apply it to the each point of the 2D PCA.
    """

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_title(r"Coloring each point with the combination of the last four features", fontsize=20)

    #===Generate color space from data
    cyan = data_matrix[:,4]
    magenta = data_matrix[:,5]
    yellow = data_matrix[:,6]
    key = data_matrix[:,7]
    colors = np.ones((x2d.shape[0],4))
    colors[:,0] = (1 -  cyan)*(1 - key)
    colors[:,1] = (1 - magenta)*(1 - key)
    colors[:,2] = (1 - yellow)*(1 - key)

    min = np.min(colors[:,0])
    max = np.max(colors[:,0])
    colors[:,0] = (colors[:,0] - min)/(max - min)
    min = np.min(colors[:,1])
    max = np.max(colors[:,1])
    colors[:,1] = (colors[:,1] - min)/(max - min)
    min = np.min(colors[:,2])
    max = np.max(colors[:,2])
    colors[:,2] = (colors[:,2] - min)/(max - min)

    colors[:,3] = 0.7

    sc = ax.scatter(x2d[:,0], x2d[:,1], c=colors, s=10)

    fig.tight_layout()
    if savefig != "":
        fig.savefig(savefig, dpi=dpi)

    plt.show()

rank = 2 #from estimate_rank
min_samples = 12
eps = 0.30

names, data_matrix, energies, not_normalized_data = prepare_data('../data/', False, 2)
print("Data loaded")

x2d, labels = PCA_DBSCAN(data_matrix, rank, eps, min_samples)
print("DBSCAN")

#scatter_plot_energies(x2d, energies, savefig="../plots/2D_PCA_all_energies.png", x1_sub=0.31, x2_sub=0.85, y1_sub=-0.87, y2_sub=-0.54)

plot_cluster_ligand_count(x2d, energies, labels, names, savefigs="../plots/ligand_count_cluster_best")

#scatter_plot_molecular_weight(x2d, not_normalized_data, "../plots/molecular_weight.png")

#plot_combination_colors(x2d, data_matrix, savefig="../plots/combination.png")
