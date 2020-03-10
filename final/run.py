# Internal libraries
import sys
import argparse

# Own libraries
import main

def parse_options():
    """Parses the few command-line options.

    Returns:
        option_string (str): The option string that is passed to the real program
        usetex (bool): The boolean whether the TeX typesetting system should be used.
    """
    parser = argparse.ArgumentParser(description='Quantum Machine Learning Standards')
    parser.add_argument('-N', '--no-plot', action='store_true', help='If specified, data plots will be generated and saved to files. Default: Plot.')
    parser.add_argument('-S', '--show', action='store_const', const=' --show', help='If specified, data plots will be shown on the run. This will require user interaction (closing windows) during the run. Default: False.')
    parser.add_argument('--usetex', action='store_const', const=' --usetex', help='Use TeX for figures. This requires a suffcient TeX distribution')

    args = parser.parse_args()

    option_string = ''
    option_string += ' --plot' if not args.no_plot else ''

    option_string += args.show if args.show else ''

    print('Given options: ' + option_string)

    return option_string, args.usetex

def run(argv):
    """The run function that runs the project in a way that it produces all our results.

    Args:
        argv (list): The list of command-line arguments given
    """
    # Parse arguments for plotting
    option_string, usetex = parse_options()

    # Rank estimation with default values
    main.main('main.py {} rankestimation'.format(option_string).split(), module = True)

    # k-distances with default values
    main.main('main.py {} kdistance'.format(option_string).split(), module = True)

    # PCA and DBSCAN with default values
    main.main('main.py pcadbscan'.split(), module = True)

    # Plotting with default values
    # This is only possible if a sufficient TeX distribution is installed
    # and can be called by this program resp. matplotlib
    if usetex:
        main.main('main.py {} --usetex plot --plots energies ligands molecularweight combination'.format(option_string).split(), module = True)

if __name__ == '__main__':
    run(sys.argv)
