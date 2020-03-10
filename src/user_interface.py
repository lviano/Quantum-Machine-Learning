import sys
import argparse

def print_error_and_exit(exit_code = 2, error_message = 'An error occured. Exiting...'):
    print(error_message, file=sys.stderr)
    exit(exit_code)

def read_command_line_options(argv):
    parser = argparse.ArgumentParser(description='Quantum machine learning')
    parser.add_argument('-p', '--path', default='../data/', help='The path to the data folder')
    parser.add_argument('-f', '--file', default='proposition_of_features.txt', help='The file that contains the proposition of features')
    parser.add_argument('-T', '--file-type', default='pdf', choices=['png', 'pdf', 'ps', 'eps', 'svg'], help='Specifies the file type of the figures. Only used if -P or --plot is specified. Must be one of ["png", "pdf", "ps", "eps", "svg"].')
    parser.add_argument('-F', '--figure-path', default='../plots', help='Specifies the path to which the figures are saved. Only used if -P or --plot is specified.')
    parser.add_argument('-e', '--elements', nargs='*', help='The list of elements to be excluded from or used for the program')
    parser.add_argument('-r', '--remove', action='store_true', help='If specified, the given --elements list will be removed from the data. Otherwise, only the given elements will be used.')
    parser.add_argument('-A', '--add-information', action='store_true', help='Specifies whether element information from periodictable should be added to the data.')
    parser.add_argument('-i', '--information', nargs='*', help='Specifies which element information from periodictable should be added. Only considered if --add-information is given.')
    parser.add_argument('-N', '--normalize', action='store_true', help='Specifies whether the data is normalized before calculations.')
    parser.add_argument('-E', '--expand', action='store_true', help='Specifies whether the features should be expanded. Give the degree with -d POSITIVE_INTEGER (default: 2).')
    parser.add_argument('-d', '--degree', type=int, default=2, help='The maximum degree of expansion. Default: 2')
    parser.add_argument('-P', '--plot', action='store_true', help='If specified, data plots will be generated and saved to files.')
    parser.add_argument('-D', '--dpi', type=int, help='Specifies the resolution in dpi for the plots.')
    parser.add_argument('-S', '--show', action='store_true', help='If specified, data plots will be sown on the run. Only works if -S or --show is specified.')
    parser.add_argument('-R', '--rank-estimation', default='Wold', choices=['IndividualF', 'AverageF', 'Wold'], help='Specifies the type of rank estimation. Must be one of ["IndividualF", "AverageF", "Wold"].')
    parser.add_argument('--epsilon', type=float, default=3, help='The epsilon (radius of the hypersphere) for DBSCAN regions. It must be a positive float.')
    parser.add_argument('--min-samples', type=int, default=56, help='The minimum number of data points that are needed in a hypersphere around a single data point such that this point is considered attractive. It must be a positive integer.')
    parser.add_argument('-L', '--logging', default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Define the console logging level')

    args = parser.parse_args()

    # Exit if --elements is given but no elements are given
    if args.elements is not None and len(args.elements) < 1:
        print_error_and_exit(2, "-e or --elements option given. Please specify at least one element.")
    # Exit if a wrong number of degrees is given
    if args.expand and args.degree < 1:
        print_error_and_exit(2, "Please pass an appropriate feature expansion degree. The degree must be a positive integer. However, it is only sensible to use a degree greater than 1. For a degree of 1, there will be no expansion at all.", file=sys.stderr)

    # Exit if a non-positive epsilon is given
    if args.epsilon <= 0.0:
        print_error_and_exit(2, "Please enter a positive epsilon.")

    # Exit if a non-positive number of samples that must be in the epsilon-radius hypersphere around a point such that in can be considered attractive
    if args.min_samples < 1:
        print_error_and_exit(2, "Please enter a positive-integer number of samples.")

    if args.add_information and (args.information is None or len(args.information) < 1):
        print_error_and_exit(2, "-A or --add-information given. Please specify add least one dimension to be added.")

    return args
