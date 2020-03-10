# External Libraries
import numpy as np

def split(name):
    """
    Custum function to split a string of the format XX_[Number1].[Number2].xyz into an array(Number1, Number2)
    
    Args:
        name (str): name of the molecule
        
    Returns:
        array containing the two numbers present in the molecule name
    """
    array = np.array(name.split('.'))
    return np.array([array[0].split('_')[1], array[1]])

def get_ligands_array(names):
    """
    Takes the a list of molecules names and outputs an array with the left and right ligand written in the name
    
    Args:
        names (numpy array): list of the molecules names of length N
        
    Returns:
        an N*2 array where the first column contains the left ligands of the molecules and the second column the right ligands of the molecules
    """
    return np.array(list(map(split, names)), dtype=int)

def get_ligand_distribution(ligands):
    """
    Takes a list of ligands (left and right) and calculates the distribution of the left and right ligand
    
    Args:
        ligands (numpy array): an N*2 array where the first contains the left ligands of the molecules and the second column the right ligands of the molecules
        
    Returns:
        distribution of the left ligands and the right ligands
    """
    n = ligands.shape[0]

    unique_l, counts_l = np.unique(ligands[:,0], return_counts=True)
    unique_r, counts_r = np.unique(ligands[:,1], return_counts=True)
    counts_l = counts_l / n
    counts_r = counts_r / n

    left = np.array(np.dstack((unique_l, counts_l))[0])
    right = np.array(np.dstack((unique_r, counts_r))[0])
    return left, right
