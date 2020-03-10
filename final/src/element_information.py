# External libraries
import periodictable as pt
import numpy as np
from operator import attrgetter

def get_element_by_symbol(symbol):
    """Retrieves the element's periodictable object by its symbol.

    Args:
        symbol (str): The element symbol for which the element object is returned.

    Returns:
        element (object): The desired element's periodictable object.
    """
    try:
        element = pt.elements.symbol(symbol)
        return element
    except ValueError as e:
        error_exit(1, 'The element with the symbol "' + symbol + '" cannot be found.')

def create_lookup_symbol(symbol):
    """Creates a one- or two-character str containing the element symbol that is used to look the respective element's information up.

    Args:
        symbol (str): The string containing the element symbol in its 1 or two first characters.

    Returns:
        lookup_symbol (str): The element symbol string.
    """
    if symbol[1].islower():
        lookup_symbol = symbol[:2]
    else:
        lookup_symbol = symbol[0]

    return lookup_symbol

def get_element_information_for_symbols(symbols):
    """Retrieves the information for the specified elements.

    Args:
        symbols (list): The list of symbol strings for the elements.
            The strings must contain the element symbol in their first two characters.

    Returns:
        elements (list): The list of elements containing the periodictable element objects.
    """
    elements = []
    for symbol in symbols:
        lookup_symbol = create_lookup_symbol(symbol)

        elements.append(get_element_by_symbol(lookup_symbol))

    return elements

def flatten_element_information(element, dimensions = ['covalent_radius', 'covalent_radius_uncertainty', 'density', 'mass', 'number']):
    """Flattens the information given in the periodictable element object to a NumPy array.

    Args:
        element (object): The periodictable element object to be flattened.
        dimensions (list): The list of dimensions (i. e. element information) to be retained from the element object.
            Defaults to ['covalent_radius', 'covalent_radius_uncertainty', 'density', 'mass', 'number'].

    Returns:
        element_information (array): The desired information of the given element flattend to an array.
    """
    flatten = attrgetter(*dimensions)
    element_information = np.asarray(flatten(element))

    return element_information

def flatten_element_array_information(elements, dimensions = ['covalent_radius', 'covalent_radius_uncertainty', 'density', 'mass', 'number']):
    """Flattens the element information for a list of element objects.

    Args:
        elements (list): The list of element object for which flattened information is desired.
        dimensions (list): The list of dimensions (i. e. element information) to be retained from the element objects.
            Defaults to ['covalent_radius', 'covalent_radius_uncertainty', 'density', 'mass', 'number'].

    Returns:
        element_information_array (array): A NumPy 2-D array containing the desired element information.
    """
    element_information_array = []

    for element in elements:
        element_information = flatten_element_information(element, dimensions)
        element_information_array.append(element_information)

    element_information_array = np.asarray(element_information_array)

    # Reshaping the vector:
    # If only one dimension is added, we need to explicitly add the second axis
    # to make it a 2-dimensional array (for how NumPy is working).
    if len(dimensions) == 1:
        element_information_array = element_information_array[:, None]

    return element_information_array

def get_element_information_array(symbols, dimensions = ['covalent_radius', 'covalent_radius_uncertainty', 'density', 'mass', 'number']):
    """Gets flattened element information for a list of element symbols.

    Args:
        symbols (list): The list of element symbols for which information is desired.
            The strings must contain the element symbol in their first two characters.
        dimensions (list): The list of dimensions (i. e. element information) to be retained from the element objects.
            Defaults to ['covalent_radius', 'covalent_radius_uncertainty', 'density', 'mass', 'number'].

    Returns:
        element_information_array (array): A NumPy 2-D array containing the desired element information.
    """
    elements = get_element_information_for_symbols(symbols)
    element_information_array = flatten_element_array_information(elements, dimensions)

    return element_information_array

def add_element_information_to_data(symbols, x_data, dimensions = ['covalent_radius', 'covalent_radius_uncertainty', 'density', 'mass', 'number']):
    """Adds specified element information to given element information data.

    Args:
        symbols (list): The list of element symbols for which information is desired.
            The strings must contain the element symbol in their first two characters.
        x_data (array): The NumPy 2-D array containing the given element information.
        dimensions (list): The list of dimensions (i. e. element information) to be retained from the element objects.
            Defaults to ['covalent_radius', 'covalent_radius_uncertainty', 'density', 'mass', 'number'].

    Returns:
        x_extended: A NumPy 2-D array containing the extended element information.
    """
    element_information_array = get_element_information_array(symbols, dimensions)
    x_extended = np.concatenate((x_data, element_information_array), axis=1)

    return x_extended

def select_element_indices_to_remove_from_data(symbols, x_data, symbols_subset, remove = False):
    """Defines rows to delete from a list of element symbols and its corresponding data array.

    Args:
        symbols (list): The list of element symbols from which rows shall be deleted.
        x_data (array): The corresponding NumPy data array.
        symbols_subset (list): The list of symbols to be either kept in or deleted from the data and the symbols list.
        remove (bool): Specifies whether the symbols in symbols_subset should be kept in or deleted from the data and the symbol list.
            Defaults to False.

    Returns:
        rows_to_delete (list): The rows to be delted from the data and the symbol list.
    """
    rows_to_delete = []

    for i in range(symbols.shape[0]):
        if (create_lookup_symbol(symbols[i]) in symbols_subset) == remove:
            rows_to_delete.append(i)

    return rows_to_delete

def select_elements_from_data(symbols, x_data, symbols_subset, remove = False):
    """Removes rows for a set of symbols from a list of symbols and the corresponding data array.

    Args:
        symbols (list): The list of element symbols from which rows shall be deleted.
        x_data (array): The corresponding NumPy data array.
        symbols_subset (list): The list of symbols to be either kept in or deleted from the data and the symbols list.
        remove (bool): Specifies whether the symbols in symbols_subset should be kept in or deleted from the data and the symbol list.
            Defaults to False.

    Returns:
        symbols_without_elements (list): The list of element symbols that are kept.
        x_data_without_elements (array): The corresponding data array that is kept.
    """
    rows_to_delete = select_element_indices_to_remove_from_data(symbols, x_data, symbols_subset, remove)

    symbols_without_elements = np.delete(symbols, rows_to_delete, 0)
    x_data_without_elements = np.delete(x_data, rows_to_delete, 0)

    return symbols_without_elements, x_data_without_elements

def select_energies_from_data(symbols, energies, symbols_subset, remove = False):
    """Removes rows for a set of symbols from a list of symbols and the corresponding energy array.

    Args:
        symbols (list): The list of element symbols from which rows shall be deleted.
        x_data (array): The corresponding NumPy data array.
        symbols_subset (list): The list of symbols to be either kept in or deleted from the data and the symbols list.
        remove (bool): Specifies whether the symbols in symbols_subset should be kept in or deleted from the data and the symbol list.
            Defaults to False.

    Returns:
        symbols_without_elements (list): The list of element symbols that are kept.
        x_data_without_elements (array): The corresponding energy array that is kept.
    """
    symbols_without_elements, energies_without_elements = select_elements_from_data(symbols, energies, symbols_subset, remove)

    return symbols_without_elements, energies_without_elements
