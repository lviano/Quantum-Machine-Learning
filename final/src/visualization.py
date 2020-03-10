# External libraries
import os
import matplotlib.pyplot as plt

# Own libraries
import debug

def read_options(args):
    """Reads the command line options for visualization

    Args:
        args (object): command line arguments

    Returns:
        options object for visualization
    """
    options = {
        'path': args.path,
        'file_type': args.file_type,
        'figure_path': args.figure_path,
        'plot': args.plot,
        'dpi': args.dpi,
        'show': args.show,
        'figsize': args.figsize,
        'fontsize': args.fontsize,
        'colormap': args.colormap,
        'usetex': args.usetex,
        'font_family': args.tex_font_family,
        'text_size': args.tex_text_size
    }

    return options

def do_plot(options):
    """Tells whether it should be plotted during the run.

    Args:
        options (dict): The dictionary containing the given visualization options

    Returns:
        plot (bool): Plot if True, otherwise do not plot
    """
    plot = options.get('plot', False)

    return plot

def get_option(options, option_key, default = None):
    """Gives an option value for a key of the visualization-options dictionary.
        A default value can be specified for the case that the key is not present in the dicitonary.
    """
    option = options.get(option_key, default)

    return option

def get_visualize_options(options):
    """Returns the option that were specified for the visualize function from the
        visualization options dictionary. Gives a default value for not given values.
    """
    path = os.path.normpath(options.get('figure_path', '../plots'))
    file_type = options.get('file_type', 'pdf')
    dpi = options.get('dpi', 300)
    show = options.get('show', False)
    close = options.get('close', True)

    return path, file_type, dpi, show, close

def visualize(figure, file_name, options):
    """Visualizes a figure. I. e. saves it as file and shows it if specified.
        Normally, closes the figure.

    Args:
        figure (object): A matplotlib figure object to be visualized
        file_name (str): The name of the generated file
        options (dict): The visualization options that have been specified
    """
    # Get options or fall back to default options
    path, file_type, dpi, show, close = get_visualize_options(options)

    # Create destination path as needed
    os.makedirs(path, exist_ok=True)

    debug.log('Saving figure to path: "' + str(os.path.abspath(path)) + '".')

    # Save figure to file
    figure.savefig(os.path.join(path, file_name + '.' + file_type), dpi=dpi, format=file_type)

    # Show figure if specified
    if show:
        plt.show(figure)

    # Close figure if specified
    if close:
        plt.close(figure)
