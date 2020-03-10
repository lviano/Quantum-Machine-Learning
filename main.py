# External libraries
import sys
sys.path.insert(0, 'src')

# Own libraries
import cluster_analysis as ca
import debug
import element_information as ei
import model_validation as mv
import load_data as ld
import option_interface as oi
import plots
import visualization as viz

def load(args):
    """Load the needed data according to the arguments given.
        If needed, given elements are removed or as only ones preserved.

    Args:
        args (object): An argparse Namespace that holds the command-line arguments as its attributes.

    Returns:
        names (array): The names of the data in the rows
        x (array): The data matrix
    """
    names = None
    x = None

    if args.information is None:
        names, x = ld.load_features(args.path, args.file, normalize = args.no_normalize, add_information = args.add_information, expand = args.expand, degree = args.degree, dimensions = args.information)
    else:
        names, x = ld.load_features(args.path, args.file, normalize = args.no_normalize, add_information = args.add_information, expand = args.expand, degree = args.degree)

    if args.elements is not None:
        names, x = ei.select_elements_from_data(names, x, args.elements, args.remove)

    return names, x

def main(argv, module = False):
    """The main functionality of the program. It processes the given arguments and
        then calls the required functionality given by the other modules.

    Args:
        argv (list): The argument vector with which the module is called. I. e.
            the command-line arguments if called from command line or a list of strings
            if called as a module.
        module (bool): Specifies whether the module was called from command line or by another module.
            This allows to adjust the argv arguments.
    """
    # Read command line options
    args = oi.read_options(argv, module = module)

    # Initialize logging
    debug.init_log(sys.stderr, args.logging)

    # Extract visualization options from options
    visualization_options = viz.read_options(args)

    if args.command == 'rankestimation':
        names, x = load(args)

        rank = mv.estimate_rank(x, visualization_options, num_seeds = args.num_seeds, k_fold_rows = args.k_fold_rows, k_fold_cols = args.k_fold_cols)

    elif args.command == 'kdistance':
        names, x = load(args)

        x_pca = ca.pca(x, args.rank)
        mv.k_distance(x_pca, visualization_options)

    elif args.command == 'pcadbscan':
        names, x = load(args)

        x2d, labels = ca.pca_dbscan_projection(x, args.rank, args.epsilon, args.min_samples, projection_dimension = args.projection_dimension)

    elif args.command == 'plot':
        plots.plot_visualization(args.path, args.file, args.energies, args.plots, visualization_options, normalize = args.no_normalize, remove = args.remove, elements = args.elements)

    else:
        debug.error_exit(1, 'Nothing to do. Exiting...')

if __name__ == '__main__':
    main(sys.argv)
