# External libraries
import argparse

def check(args):
    """Check arguments for (some) user mistakes. Errors if a wrong option is given.

    Args:
        args (object): The argparse Namespace object containing the given program arguments as attributes
    """
    # Exit if --elements is given but no elements are given
    if args.elements is not None and len(args.elements) < 1:
        print_error_and_exit(2, "-e or --elements option given. Please specify at least one element.")

    # Exit if a wrong number of degrees is given
    if args.expand and args.degree < 1:
        print_error_and_exit(2, "Please pass an appropriate feature expansion degree. The degree must be a positive integer. However, it is only sensible to use a degree greater than 1. For a degree of 1, there will be no expansion at all.", file=sys.stderr)

    # Exit if a wrong number of degrees is given
    if args.expand and args.degree < 1:
        print_error_and_exit(2, "Please pass an appropriate feature expansion degree. The degree must be a positive integer. However, it is only sensible to use a degree greater than 1. For a degree of 1, there will be no expansion at all.", file=sys.stderr)

    # Exit if a non-positive epsilon is given
    if args.epsilon is not None and args.epsilon <= 0.0:
        print_error_and_exit(2, "Please enter a positive epsilon.")

    # Exit if a non-positive number of samples that must be in the epsilon-radius hypersphere around a point so that in can be considered attractive is given
    if args.min_samples < 1:
        print_error_and_exit(2, "Please enter a positive-integer number of samples.")

    # Exit if information shall be added but no information is specified
    if args.add_information and (args.information is None or len(args.information) < 1):
        print_error_and_exit(2, "-A or --add-information given. Please specify add least one dimension to be added.")

def read_options(argv, module = False):
    """Read command-line arguments. Checks for some user mistakes.

    Args:
        argv (list): The argument vector
        module (bool): Whether the calling module was called from command line or from a module

    Returns:
        args (object): An object containing the parsed command-line arguments as attributes
    """
    # Create parser with standard options
    parser = argparse.ArgumentParser(description='Quantum Machine Learning')
    parser.add_argument('-p', '--path', default='./data', help='The path to the data folder. Default: %(default)s.')
    parser.add_argument('-f', '--file', default='proposition_of_features.txt', help='The file that contains the proposition of features. Default: %(default)s.')
    parser.add_argument('-e', '--energies', default='comp_BoB_reordered_energies.txt', help='The file that contains the energies. Default: %(default)s.')

    parser.add_argument('-t', '--file-type', default='png', choices=['png', 'pdf', 'ps', 'eps', 'svg'], help='Specifies the file type of the figures. Only used if -P or --plot is specified. Must be one of ["png", "pdf", "ps", "eps", "svg"].')
    parser.add_argument('-F', '--figure-path', default='./plots', help='Specifies the path to which the figures are saved. Only used if -P or --plot is specified. Default: %(default)s.')
    parser.add_argument('-P', '--plot', action='store_true', help='If specified, data plots will be generated and saved to files. Default: False.')
    parser.add_argument('-D', '--dpi', type=int, default=300, help='Specifies the resolution in dpi for the plots. Default: %(default)s.')
    parser.add_argument('-S', '--show', action='store_true', help='If specified, data plots will be shown on the run. This will require user interaction (closing windows) during the run. Default: False.')
    parser.add_argument('--figsize', type=tuple, default=(10, 8), help='The size of the figures. Default: %(default)s.')
    parser.add_argument('--fontsize', type=int, default=24, help='The font size in the figures. Default: %(default)s.')
    parser.add_argument('--colormap', type=str, default='gist_rainbow', help='The standard colormap for the figures. Default: %(default)s.')
    parser.add_argument('--usetex', action='store_true', help='Use TeX for figure text. This requires a sufficient TeX distribution. It is needed for some figures. Default: False.')
    parser.add_argument('--tex-font-family', type=str, default='serif', help='The TeX font for figure text. Only used if --usetex is specified. Default: %(default)s.')
    parser.add_argument('--tex-text-size', type=str, default=20, help='The TeX text size for figure text. Only used if --usetex is specified. Default: %(default)s.')

    parser.add_argument('-E', '--elements', nargs='*', help='The list of elements to be excluded from or used for the program.')
    parser.add_argument('-r', '--remove', action='store_true', help='If specified, the given --elements list will be removed from the data. Otherwise, only the given elements will be used. Default: False.')
    parser.add_argument('-A', '--add-information', action='store_true', help='Specifies whether element information from periodictable should be added to the data. Default: False.')
    parser.add_argument('-i', '--information', nargs='*', help='Specifies which element information from periodictable should be added. Only considered if -A or --add-information is given.')
    parser.add_argument('-N', '--no-normalize', action='store_false', help='Specifies whether the data is normalized before calculations. Default: Normalize.')
    parser.add_argument('--expand', action='store_true', help='Specifies whether the features should be expanded. Give the degree with -d POSITIVE_INTEGER (default: 2). Default: False.')
    parser.add_argument('-d', '--degree', type=int, default=2, help='The maximum degree of expansion. Default: %(default)s.')

    parser.add_argument('-L', '--logging', default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Define the console logging level. Default: %(default)s.')

    # Create subparsers
    subparsers = parser.add_subparsers(dest='command')

    # Subparser for rank estimation
    parser_rank_estimation = subparsers.add_parser('rankestimation')
    parser_rank_estimation.add_argument('-N', '--num-seeds', type=int, default=1, help='The number of times the gabriel cross validation scheme has to be repeated. Default: %(default)s.')
    parser_rank_estimation.add_argument('-R', '--k-fold-rows', type=int, default=23, help='The number of row-indices subsets to be considered. Default: %(default)s.')
    parser_rank_estimation.add_argument('-C', '--k-fold-cols', type=int, default=8, help='The number of column-indices subsets to be considered. Default: %(default)s.')

    # Subparser for k-distance plot
    parser_k_distance = subparsers.add_parser('kdistance')
    parser_k_distance.add_argument('-R', '--rank', type=int, default=2, help='The desired rank for the PCA. Default: %(default)s.')

    # Subparser for PCA and DBSCAN
    parser_pca_dbscan = subparsers.add_parser('pcadbscan')
    parser_pca_dbscan.add_argument('--epsilon', type=float, default=0.3, help='The epsilon (radius of the hypersphere) for DBSCAN regions. It must be a positive float. Default: %(default)s.')
    parser_pca_dbscan.add_argument('--min-samples', type=int, default=12, help='The minimum number of data points that are needed in a hypersphere around a single data point such that this point is considered attractive. It must be a positive integer. Default: %(default)s.')
    parser_pca_dbscan.add_argument('-R', '--rank', type=int, default=2, help='The desired rank for the PCA. Default: %(default)s.')
    parser_pca_dbscan.add_argument('--projection-dimension', type=int, default=2, help='The number of dimensions to which the data is projected for visualization. Default: %(default)s. Do not change this value if you do not really know what you are doing or do not provide proper means of visualization!')

    # Subparser for the plots
    parser_plot = subparsers.add_parser('plot')
    parser_plot.add_argument('--plots', nargs='+', default='energies', choices=['energies', 'ligands', 'molecularweight', 'combination'], help='Specifies the wanted plots. Default: %(default)s.')

    # Parse arguments
    if module: # If called as a module
        args = parser.parse_args(argv[1:])
    else: # If called from command line (standard)
        args = parser.parse_args()

    return args
