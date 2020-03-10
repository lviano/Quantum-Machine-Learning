# External libraries
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

def pca(x, n_components):
    """Applies PCA on the data matrix x (alias function of scikit PCA)

    Args:
        x (numpy array): normalized data matrix
        n_components (int): desired rank for the PCA

    Returns:
        x_transformed: orthogonal transformed data with n_components
    """
    x_transformed = PCA(n_components = int(n_components)).fit_transform(x)

    return x_transformed

def dbscan(x, epsilon, min_samples):
    """Performs a DBSCAN on the data x to try to cluster the data (alias function of scikit DBSCAN)

    Args:
        x (numpy array): data to cluster
        epsilon (float): radius of the hypersphere for finding the clusters
        min_samples (int): minimum number of samples to be considered inside a cluster

      Returns:
        clustering: array of labeled data for each point of the data matrix x
    """
    clustering = DBSCAN(eps = epsilon, min_samples = min_samples).fit(x)

    return clustering

def pca_dbscan(x, rank, epsilon, min_samples):
    """Applies a PCA and then finds the clusters with DBSCAN

    Args:
        x (numpy array): data matrix
        rank (int): desired rank for the PCA
        epsilon (float): radius of the hypersphere for finding the clusters
        min_samples (int): minimum number of samples to be considered inside a cluster

    Returns:
        x_transformed:  orthogonal transformed data with rank
        clustering: array of labeled data for each point of the data matrix x
    """
    x_transformed = pca(x, rank)
    clustering = dbscan(x_transformed, epsilon, min_samples)

    return x_transformed, clustering

def pca_dbscan_projection(x, rank, epsilon, min_samples, projection_dimension = 2):
    """Performs a PCA with the given rank as number of components on the data array x.
        Then, finds a clustering with DBSCAN. Returns a projection to the specified dimension.

    Args:
        x (array): normalized 2-D data array
        rank (int): desired rank for the PCA
        epsilon (float): epsilon value for DBSCAN (i. e. the hypersphere radius)
        min_samples (int): minimal amount of samples per cluster for DBSCAN
        projection_dimension (int): dimension to which the clustering shall be projected.
            This is useful for visualization, e. g. with 2 dimensions. Defaults to 2.

    Returns:
        x_projected: dimensional projection of the clustered data
        labels: labels defined by DBSCAN for each point that can be used to select indiviual clusters
    """
    x_transformed, clustering = pca_dbscan(x, rank, epsilon, min_samples)

    labels = clustering.labels_
    x_projected = pca(x, projection_dimension)

    return x_projected, labels
