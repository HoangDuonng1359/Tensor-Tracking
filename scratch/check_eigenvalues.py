import numpy as np
from analysis.paper_alignment import load_condition_stream
from algorithms.common import concatenate_mode_unfoldings

stream = load_condition_stream("incorrect")
# Look at the window around Frame 128 (ERN)
window = stream[124:132] # alpha=8
mode = 0
unfolded = concatenate_mode_unfoldings(window, mode)
cov = (unfolded @ unfolded.T) / unfolded.shape[1]
evals = np.linalg.eigvalsh(cov)
evals = np.sort(evals)[::-1]
print(f"Eigenvalues of Mode 0 covariance at ERN window: {evals[:10]}")
