import numpy as np
import numpy.typing as npt
from typing import Tuple
from .fiedler import recursive_repartitioning

def fcca_on_lowrank_interval(
    lowrank_stream: npt.NDArray[np.float64], 
    start_idx: int, 
    end_idx: int,
    threshold: float = 0.15
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """
    Applies the Fiedler Consensus Clustering Approach (FCCA) with 
    Recursive Repartitioning to detect community structures.
    """
    lower = max(0, int(start_idx))
    upper = min(lowrank_stream.shape[0] - 1, int(end_idx))
    if lower > upper:
        return np.zeros(lowrank_stream.shape[1], dtype=np.int64), np.zeros((lowrank_stream.shape[1], lowrank_stream.shape[1]), dtype=np.float64)

    # `compute_intervals()` stores inclusive end indices, so FCCA must include
    # both boundaries when slicing interval data from the tracked tensor stream.
    interval_data = lowrank_stream[lower:upper + 1]
    delta_t, n_nodes, _, n_subs = interval_data.shape
    total_matrices = delta_t * n_subs
    W = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    
    if total_matrices == 0:
        return np.zeros(n_nodes, dtype=np.int64), W

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
    final_partition = np.zeros(n_nodes, dtype=np.int64)
    for idx, comm in enumerate(final_communities):
        for node in comm:
            final_partition[node] = idx
            
    return final_partition, W
