import numpy as np

def split(name):
    array = np.array(name.split('.'))
    return np.array([array[0].split('_')[1], array[1]])

def get_ligands_array(names):
    """
    """
    return np.array(list(map(split, names)), dtype=int)

def get_ligand_distribution(ligands):
    """
    """
    n = ligands.shape[0]

    unique_l, counts_l = np.unique(ligands[:,0], return_counts=True)
    unique_r, counts_r = np.unique(ligands[:,1], return_counts=True)
    counts_l = counts_l / n
    counts_r = counts_r / n

    left = np.array(np.dstack((unique_l, counts_l))[0])
    right = np.array(np.dstack((unique_r, counts_r))[0])
    return left, right
