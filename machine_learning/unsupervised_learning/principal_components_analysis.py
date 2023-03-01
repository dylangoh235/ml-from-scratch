import numpy as np

class PCA:
    """
    Principal Components Analysis (PCA) is a method for reducing the feature space of a data matrix.

    Parameters:
    -----------
    r: int
        The number of principal components to keep.
    """
    def __init__(self, r):
        self.r = r

    def transform(self, X):
        """Using SVD to transform X to a lower dimensional space of rank r. 
        Ekward-young theorem says that the first r columns of U are the eigenvectors of XX^T which are the principal components of X.
        """
        
        U, S, VT = np.linalg.svd(X, full_matrices=False)
        self.components_ = VT
        self.explained_variance_ratio_ = (S ** 2) / np.sum(S ** 2)
        return U[:, :self.r] @ np.diag(S[:self.r])
        
