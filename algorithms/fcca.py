import numpy as np
from scipy.linalg import eigh
from typing import Tuple, List

def compute_fiedler_vector(adj_matrix: np.ndarray) -> np.ndarray:
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
        return np.zeros(n)

def recursive_repartitioning(adj_matrix: np.ndarray, nodes: List[int], min_size: int = 4) -> List[List[int]]:
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

def fcca_on_lowrank_interval(
    lowrank_stream: np.ndarray, 
    start_idx: int, 
    end_idx: int,
    threshold: float = 0.15
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies the Fiedler Consensus Clustering Approach (FCCA) with 
    Recursive Repartitioning to detect community structures.
    """
    lower = max(0, int(start_idx))
    upper = min(lowrank_stream.shape[0] - 1, int(end_idx))
    if lower > upper:
        return np.zeros(lowrank_stream.shape[1]), np.zeros((lowrank_stream.shape[1], lowrank_stream.shape[1]))

    # `compute_intervals()` stores inclusive end indices, so FCCA must include
    # both boundaries when slicing interval data from the tracked tensor stream.
    interval_data = lowrank_stream[lower:upper + 1]
    delta_t, n_nodes, _, n_subs = interval_data.shape
    total_matrices = delta_t * n_subs
    W = np.zeros((n_nodes, n_nodes))
    
    if total_matrices == 0:
        return np.zeros(n_nodes), W

    for t in range(delta_t):
        for s in range(n_subs):
            adj_matrix = interval_data[t, :, :, s].copy()
            adj_matrix[adj_matrix < threshold] = 0
            
            # Use recursive partitioning for k > 2
            communities = recursive_repartitioning(adj_matrix, list(range(n_nodes)))
            
            # Update co-occurrence matrix W
            for comm in communities:
                for i in comm:
                    for j in comm:
                        W[i, j] += 1.0
                            
    W /= total_matrices
    
    # Final consensus clustering on W
    n_nodes = W.shape[0]
    final_communities = recursive_repartitioning(W, nodes=list(range(n_nodes)), min_size=4)
    
    # Convert list of lists to partition array
    final_partition = np.zeros(n_nodes, dtype=int)
    for idx, comm in enumerate(final_communities):
        for node in comm:
            final_partition[node] = idx
            
    return final_partition, W
