import numpy as np
import pandas as pd
from algorithms.ho_rlsl import HORLSLRunner, default_config_for_condition as horlsl_cfg
from fcca import fcca_on_lowrank_interval
from algorithms.common import convert_subject_tensor_to_stream
import networkx as nx

def calculate_graph_metrics(W, top_percentile=0.2):
    """
    Quantifies the graph properties of a consensus matrix using 
    proportional thresholding (keeping only top edges).
    """
    n = W.shape[0]
    # Flatten and get threshold for top X% edges
    flat_w = W[np.triu_indices(n, k=1)]
    threshold = np.percentile(flat_w, 100 * (1 - top_percentile))
    
    # Adjacency matrix (weighted)
    adj = W.copy()
    adj[adj < threshold] = 0
    
    G = nx.from_numpy_array(adj)
    
    # Metrics
    metrics = {}
    metrics['threshold'] = threshold
    metrics['density'] = nx.density(G)
    metrics['efficiency'] = nx.global_efficiency(G)
        
    # Weighted Modularity
    communities = nx.community.greedy_modularity_communities(G, weight='weight')
    metrics['modularity'] = nx.community.modularity(G, communities, weight='weight')
    metrics['num_communities'] = len(communities)
    
    # Centrality (Weighted)
    centrality = nx.degree_centrality(G) # NX degree centrality on thresholded graph
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
    metrics['top_hubs'] = [int(n) for n, c in top_nodes]
    
    return metrics

def perform_statistical_quantification():
    print("🧪 Starting Quantitative Network Analysis (HO-RLSL)...")
    
    data = np.load("data/processed_v2/03_connectivity_tensors/tensor_incorrect_4d.npy").astype(np.float32)
    stream = convert_subject_tensor_to_stream(data)
    
    runner = HORLSLRunner(horlsl_cfg())
    result = runner.run(stream)
    
    # Electrode labels for mapping
    CHANNELS = ["FP1", "F3", "F7", "FC3", "C3", "C5", "P3", "P7", "P9", "PO7", "PO3", "O1", "Oz", "Pz", "CPz", "FP2", "Fz", "F4", "F8", "FC4", "FCz", "Cz", "C4", "C6", "P4", "P8", "P10", "PO8", "PO4", "O2"]
    
    # 1. Define Phases
    phases = {
        "Pre-ERN": [40, 80],
        "ERN (Peak)": [124, 132], # Using detected window
        "Post-ERN": [180, 220]
    }
    
    summary_results = []
    
    for name, window in phases.items():
        _, W = fcca_on_lowrank_interval(result.lowrank_stream, window[0], window[1], threshold=0.15)
        m = calculate_graph_metrics(W)
        
        hub_names = [CHANNELS[i] for i in m['top_hubs']]
        
        summary_results.append({
            "Phase": name,
            "Modularity (Q)": f"{m['modularity']:.4f}",
            "Efficiency": f"{m['efficiency']:.4f}",
            "Density": f"{m['density']:.4f}",
            "Top Hubs": ", ".join(hub_names)
        })
        
    df = pd.DataFrame(summary_results)
    print("\n" + "="*80)
    print("📊 QUANTITATIVE GRAPH METRICS REPORT")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # 2. Permutation Test (Simplistic for Energy at 0ms)
    print("\n🎲 Running Permutation Test for ERN Energy Peak...")
    actual_ern_energy = np.mean(result.residual_energy[124:132])
    
    null_energies = []
    for _ in range(100):
        shuffled_energy = np.random.choice(result.residual_energy, 8)
        null_energies.append(np.mean(shuffled_energy))
    
    p_val = np.sum(np.array(null_energies) >= actual_ern_energy) / 100
    print(f"   ERN Energy (Mean): {actual_ern_energy:.6f}")
    print(f"   Permutation P-Value: {p_val:.4f} " + ("(Significant ⭐)" if p_val < 0.05 else "(Not Significant)"))

if __name__ == "__main__":
    perform_statistical_quantification()
