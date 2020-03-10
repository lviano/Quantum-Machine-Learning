# External libraries
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np

# Own libraries
import cluster_analysis as ca
import element_information as ei
import ligand_information as li
import load_data as ld
import visualization as viz

def prepare_data(path, filename_features, filename_energies, normalize = True, expand = False, degree = 2, remove = False, elements = []):
    """Loads and prepares the data necessary for the plotting

    Args:
        path (str): path containing the data
        filename_features (str): name of the features data set file
        filename_energies (str): name of file for the energies associated to the molecules
        normalize (bool): if True, normalizes the data matrix (features)
        expand (bool): if True, makes column-wise expansion of the data matrix (features)
        degree (int):  degree of expansion. Only works if expand=True
        remove (bool): if True, removes from the data the molecules containing the elements in the list elements
        elements (list): if remove=True, then all molecules elements specified in this list will be removed from the data

    Returns:
        names: names of the load molecules
        x: data matrix (features)
        energies: Delta G(Rxn A) energies of the molecules
        not_normalized_data: non normalized data matrix (features)
        min_energy: minimum present in all energies
        max_energy: maximum present in all energies

    Note: all the data is being shuffled (but preserving the order between names, x, energies and not_normalized_data)
    """
    names, x = ld.load_features(path, filename_features, expand = expand, degree = degree, normalize = normalize)
    _, not_normalized_data = ld.load_features(path, filename_features, expand = False, normalize = False)
    energies, energies_names = ld.load_energies(path, filename_energies)

    min_energy = np.min(energies)
    max_energy = np.max(energies)

    if elements is not None and len(elements) > 0:
        names, x = ei.select_elements_from_data(names, x, elements, remove = remove)
        _, not_normalized_data = ei.select_elements_from_data(names, not_normalized_data, elements, remove = remove)
        energies_names, energies = ei.select_energies_from_data(energies_names, energies, elements, remove = remove)

    indices = np.random.permutation(names.shape[0])
    return names[indices], x[indices], energies[indices], not_normalized_data[indices], min_energy, max_energy


def scatter_plot_energies(x2d, energies, visualization_options, min_energy, max_energy, x1_sub=-2.06, zoom_on_plot=True, x2_sub=-1.70, y1_sub=0.96, y2_sub=1.37):
    """Plots the 2D PCA of the data with the associated energies

    Args:
        x2d (numpy array): 2d matrix containing the points given by a 2D PCA on the data matrix
        energies (numpy array): array of all the Delta G(Rxn A) energies of the molecules
        visualization_options (dict): contains the options for visualization (plotting)
        zoom_on_plot (bool): if True, defines a region inside the actual plot on which it is possible to zoom to have better details
        min_energy (float): minimum energy present in all the energies
        max_energy (float): maximum energy present in all the energies
        x1_sub (float): top left x coordinate for defining a box on which the plot is zoomed on
        x2_sub (float): bottom right x coordinate for defining a zone on which the plot is zoomed on
        y1_sub (float): top left y coordinate for defining a box on which the plot is zoomed on
        y2_sub (float): bottom right y coordinate for defining a box on which the plot is zoomed on
    """
    # Set given plot values locally
    figsize = viz.get_option(visualization_options, 'figsize')
    fontsize = viz.get_option(visualization_options, 'fontsize')
    colormap = viz.get_option(visualization_options, 'colormap')

    # === Main plot ===
    # Title of plot
    energy_fig = plt.figure(figsize=figsize)
    ax = energy_fig.add_subplot(111)
    ax.set_title(r"2D PCA with the associated $\Delta G (\textrm{Rxn}$ $A)$ energies", fontsize=fontsize)

    scatter = ax.scatter(x2d[:,0], x2d[:,1], c=energies, cmap=colormap, alpha=0.05, s=10, vmin=min_energy, vmax=max_energy)
    colorbar = plt.colorbar(scatter, label=r"$\Delta G (\textrm{Rxn}$ $A)$ [kcal/mol]")
    colorbar.set_alpha(0.5)
    colorbar.draw_all()

    if zoom_on_plot:
        # === Zoomed plot ===
        axin = zoomed_inset_axes(ax, 4, loc=2)
        axin.scatter(x2d[:,0], x2d[:,1], c=energies, cmap=colormap, alpha=0.1, s=10, vmin=min_energy, vmax=max_energy)
        # subregion of the original image
        axin.set_xlim(x1_sub, x2_sub)
        axin.set_ylim(y1_sub, y2_sub)

        plt.xticks(visible=False)
        plt.yticks(visible=False)

        # Draw a frame around the subplot
        mark_inset(ax, axin, loc1=1, loc2=3, fc="none", ec="0.5")

    plt.tight_layout()

    if viz.do_plot(visualization_options):
        viz.visualize(energy_fig, 'energies', visualization_options)

def plot_cluster_ligand_count(x2d, energies, labels, names, visualization_options):
    """Plots the histogram of the ligands (left and right) of all individual clusters found by DBSCAN

    Args:
        x2d (numpy array): 2d matrix containing the points given by a 2D PCA on the data matrix
        energies (numpy array): array of all the Delta G(Rxn A) energies of the molecules
        labels (numpy array): array containing the labels of each molecule that has been assigned by DBSCAN
        visualization_options (dict): contains the options for visualization (plotting)
    """
    # Set given plot values locally
    figsize = viz.get_option(visualization_options, 'figsize')
    fontsize = viz.get_option(visualization_options, 'fontsize')

    #For the colorbar, find the min and max energies
    min = np.min(energies)
    max = np.max(energies)

    unique_labels = set(labels)
    ligands = li.get_ligands_array(names)

    _, counts_left = np.unique(ligands[:,0], return_counts=True)
    _, counts_right = np.unique(ligands[:,1], return_counts=True)

    total_ligands_nbr = (counts_left + counts_right)[0]

    for i, k in enumerate(unique_labels):
        if k == -1:
            continue

        indices = (labels == k)
        mask = np.ones(labels.shape, bool)
        mask[indices] = False

        unique_ligands = np.unique(ligands[:,0])

        # === Count of the left and right ligand species ===
        unique_left, counts_left = np.unique(ligands[:,0][indices], return_counts=True)
        unique_right, counts_right = np.unique(ligands[:,1][indices], return_counts=True)

        counts = np.zeros(unique_ligands.shape)
        counts[unique_left] = counts_left
        counts[unique_right] += counts_right
        counts /= total_ligands_nbr

        # === Scatter plot ===
        ligand_cluster_fig = plt.figure(figsize=figsize)
        ax = ligand_cluster_fig.add_subplot(211)
        ax.scatter(x2d[mask,0], x2d[mask,1], c='gray', s=10, alpha=0.01)
        ax.set_title(r"Selected cluster", fontsize=fontsize)
        scatter = ax.scatter(x2d[indices,0], x2d[indices,1], c=energies[indices], cmap='hsv', s=10, alpha=0.05, vmin=min, vmax=max)
        colorbar = plt.colorbar(scatter, label=r"$\Delta G (\textrm{Rxn}$ $A)$ [kcal/mol]")
        colorbar.set_alpha(0.5)
        colorbar.draw_all()

        plt.subplots_adjust(hspace = 0.3)

        # === Histogram ===
        ax2 = ligand_cluster_fig.add_subplot(212)
        ax2.set_title(r"Ligands count in selected molecules cluster", fontsize=fontsize)
        ax2.bar(unique_ligands, counts)
        ax2.set_xlabel(r'Ligand species number')
        ax2.set_ylabel(r'Percentage of all ligands')
        ax2.set_xlim(0, 90)
        ligand_cluster_fig.tight_layout()

        if viz.do_plot(visualization_options):
            viz.visualize(ligand_cluster_fig, 'ligand_clusters' + str(k), visualization_options)

def scatter_plot_molecular_weight(x2d, data_matrix, visualization_options):
    """Plots the 2D PCA with the molecules colored by their molecular weight

    Args:
        x2d (numpy array): 2d matrix containing the points given by a 2D PCA on the data matrix
        data_matrix (numpy array): non normalized data matrix of all the features
        visualization_options (dict): contains the options for visualization (plotting)
    """
    molecular_weight_fig = plt.figure(figsize=viz.get_option(visualization_options, 'figsize'))
    ax = molecular_weight_fig.add_subplot(111)

    ax.set_title(r"2D PCA with molecules colored by their molecular volume", fontsize=viz.get_option(visualization_options, 'fontsize'))
    scatter = ax.scatter(x2d[:,0], x2d[:,1], c=data_matrix[:,2], alpha=0.05, s=10, cmap=viz.get_option(visualization_options, 'colormap'))
    colorbar = plt.colorbar(scatter, label=r"Molecular volume [m$^3$/mol]")
    colorbar.set_alpha(0.5)
    colorbar.draw_all()

    molecular_weight_fig.tight_layout()

    if viz.do_plot(visualization_options):
        viz.visualize(molecular_weight_fig, 'molecular_weights', visualization_options)

def plot_combination_colors(x2d, data_matrix, visualization_options):
    """Tries to color code the last four features in the data set and apply it to the each point of the 2D PCA.

    Args:
        x2d (numpy array): 2d matrix containing the points given by a 2D PCA on the data matrix
        data_matrix (numpy array): data matrix of all the features
        visualization_options (dict): contains the options for visualization (plotting)
    """

    combination_colors_fig = plt.figure(figsize=viz.get_option(visualization_options, 'figsize'))
    ax = combination_colors_fig.add_subplot(111)

    ax.set_title(r"Coloring each point with the combination of the last four features", fontsize=20)

    # === Generate color space from data ===
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

    scatter = ax.scatter(x2d[:,0], x2d[:,1], c=colors, s=10)

    combination_colors_fig.tight_layout()

    if viz.do_plot(visualization_options):
        viz.visualize(combination_colors_fig, 'combination_colors', visualization_options)

def plot_visualization(path, filename_features, filename_energies, plots, visualization_options, normalize = True, remove = False, elements = [], expand = False, degree = 2, rank = 2, min_samples = 12, epsilon = 0.3):
    """Function for choosing which plot to make that are used in the final report or the plot appendix

    Args:
        path (str): path containing the data
        filename_features (str): name of the features data set file
        filename_energies (str): name of file for the energies associated to the molecules
        plots (list): list of desired plots (energies, ligands, molecularweight or combination)
        visualization_options (dict): contains the options for visualization (plotting)
        normalize (bool): if True, normalizes the data matrix (features)
        expand (bool): if True, makes column-wise expansion of the data matrix (features)
        degree (int):  degree of expansion. Only works if expand=True
        remove (bool): if True, removes from the data the molecules containing the elements in the list elements
        elements (list): if remove=True, then all molecules elements specified in this list will be removed from the data
    """
    # Load data
    names, x, energies, not_normalized_data, min, max = prepare_data(path, filename_features, filename_energies, expand = expand, degree = degree, remove = remove, elements = elements, normalize = normalize)

    x2d, labels = ca.pca_dbscan_projection(x, rank, epsilon, min_samples, projection_dimension = 2)

    # Plotting set-up
    if viz.do_plot(visualization_options):
        plt.rc('text', usetex = viz.get_option(visualization_options, 'usetex', True))
        plt.rc('font', family = viz.get_option(visualization_options, 'font_family', 'serif'), size = viz.get_option(visualization_options, 'text_size', 20))

        if viz.get_option(visualization_options, 'figsize') is None:
            visualization_options['figsize'] = (10, 8)
        if viz.get_option(visualization_options, 'fontsize') is None:
            visualization_options['fontsize'] = 24
        if viz.get_option(visualization_options, 'colormap') is None:
            visualization_options['colormap'] = 'gist_rainbow'
        if viz.get_option(visualization_options, 'dpi') is None:
            visualization_options['dpi'] = 300

    if 'energies' in plots:
    #scatter_plot_energies(x2d, energies, savefig="../plots/2D_PCA_all_energies.png", x1_sub=0.31, x2_sub=0.85, y1_sub=-0.87, y2_sub=-0.54)
        scatter_plot_energies(x2d, energies, visualization_options, min, max, zoom_on_plot=False, x1_sub=0.31, x2_sub=0.85, y1_sub=-0.87, y2_sub=-0.54)

    if 'ligands' in plots:
    #plot_cluster_ligand_count(x2d, energies, labels, names, savefigs="../plots/ligand_count_cluster_best")
        plot_cluster_ligand_count(x2d, energies, labels, names, visualization_options)

    if 'molecularweight' in plots:
    #scatter_plot_molecular_weight(x2d, not_normalized_data, "../plots/molecular_weight.png")
        scatter_plot_molecular_weight(x2d, not_normalized_data, visualization_options)

    if 'combination' in plots:
        plot_combination_colors(x2d, x, visualization_options)
