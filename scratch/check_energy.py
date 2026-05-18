import numpy as np
import matplotlib.pyplot as plt
from analysis.paper_alignment import load_condition_stream
from core.config import config

stream = load_condition_stream("incorrect")
energy = np.mean(stream**2, axis=(1, 2, 3))

plt.figure(figsize=(10, 4))
plt.plot(np.arange(len(energy)) - 128, energy)
plt.axvline(0, color='red', linestyle='--')
plt.title("Raw Tensor Energy over Time")
plt.xlabel("Time (ms)")
plt.ylabel("Mean Squared Energy")
plt.savefig(config.paths.OUTPUTS_DIR / "energy_profile.png")
print(f"Energy at start (T=0): {energy[0]:.6f}")
print(f"Energy at ERN (T=128): {energy[128]:.6f}")
