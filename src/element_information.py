import periodictable as pt
import numpy as np
from operator import attrgetter

def get_element_by_symbol(symbol):
    try:
        element = pt.elements.symbol(symbol)
        return element
    except ValueError as e:
        return e

def create_lookup_symbol(symbol):
    if symbol[1].islower():
        lookup_symbol = symbol[:2]
    else:
        lookup_symbol = symbol[0]

    return lookup_symbol

def get_element_information_for_symbols(symbols):
    elements = []
    for symbol in symbols:
        lookup_symbol = create_lookup_symbol(symbol)

        elements.append(get_element_by_symbol(lookup_symbol))

    return elements

def flatten_element_information(element, dimensions = ['covalent_radius', 'covalent_radius_uncertainty', 'density', 'mass', 'number']):
    flatten = attrgetter(*dimensions)
    element_information = np.asarray(flatten(element))

    return element_information

def flatten_element_array_information(elements, dimensions = ['covalent_radius', 'covalent_radius_uncertainty', 'density', 'mass', 'number']):
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
    elements = get_element_information_for_symbols(symbols)
    element_information_array = flatten_element_array_information(elements, dimensions)

    return element_information_array

def add_element_information_to_data(symbols, x_data, dimensions = ['covalent_radius', 'covalent_radius_uncertainty', 'density', 'mass', 'number']):
    element_information_array = get_element_information_array(symbols, dimensions)
    x_extended = np.concatenate((x_data, element_information_array), axis=1)

    return x_extended

def select_element_indices_to_remove_from_data(symbols, x_data, symbols_subset, remove = False):
    rows_to_delete = []

    for i in range(symbols.shape[0]):
        if (create_lookup_symbol(symbols[i]) in symbols_subset) == remove:
            rows_to_delete.append(i)

    return rows_to_delete

def select_elements_from_data(symbols, x_data, symbols_subset, remove = False):
    rows_to_delete = select_element_indices_to_remove_from_data(symbols, x_data, symbols_subset, remove)

    symbols_without_elements = np.delete(symbols, rows_to_delete, 0)
    x_data_without_elements = np.delete(x_data, rows_to_delete, 0)

    return symbols_without_elements, x_data_without_elements

def select_energies_from_data(symbols, energies, symbols_subset, remove = False):
    rows_to_delete = select_element_indices_to_remove_from_data(symbols, energies, symbols_subset, remove)

    symbols_without_elements = np.delete(symbols, rows_to_delete, 0)
    energies_without_elements = np.delete(energies, rows_to_delete, 0)

    return symbols_without_elements, energies_without_elements

# all: covalent_radius, covalent_radius_uncertainty, density, mass, number
# sparse: number_density, K_alpha, K_beta1
# too complex: crystal_structure, ion, ions, isotopes, magnetic_ff, neutron
# nonsensical: *_units, density_caveat, name, symbol, table, xray
# i don't have a clue: interatomic_distance (How can an atom have an *inter*atomic distance?),
