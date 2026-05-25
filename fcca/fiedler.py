import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh
from typing import List

def compute_fiedler_vector(adj_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Compute the Fiedler vector (eigenvector corresponding to the 2nd smallest 
    eigenvalue of the Laplacian matrix) from an adjacency matrix.
    """
    n = adj_matrix.shape[0]
    # Symmetrize just in case
    adj_matrix = (adj_matrix + adj_matrix.T) / 2.0
    
    d = np.diag(np.sum(adj_matrix, axis=1))
    laplacian = d - adj_matrix
    
    try:
        # eigh solves Hermitian eigenvalue problem, returns sorted eigenvalues
        eigenvalues, eigenvectors = eigh(laplacian)
        # 0th is the trivial zero eigenvalue, 1st is the Fiedler vector
        return eigenvectors[:, 1]
    except Exception as e:
        print(f"Eigenvalue solver error: {e}")
        return np.zeros(n, dtype=np.float64)

def recursive_repartitioning(adj_matrix: npt.NDArray[np.float64], nodes: List[int], min_size: int = 4) -> List[List[int]]:
    """
    Recursively partitions nodes using Fiedler vector to find k > 2 communities.
    """
    if len(nodes) < min_size * 2:
        return [nodes]
        
    sub_adj = adj_matrix[np.ix_(nodes, nodes)]
    f_vec = compute_fiedler_vector(sub_adj)
    
    group1 = [nodes[i] for i, v in enumerate(f_vec) if v > 0]
    group2 = [nodes[i] for i, v in enumerate(f_vec) if v <= 0]
    
    if not group1 or not group2:
        return [nodes]
        
    return recursive_repartitioning(adj_matrix, group1, min_size) + \
           recursive_repartitioning(adj_matrix, group2, min_size)
