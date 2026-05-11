import mne
import numpy as np
from pathlib import Path
from typing import List
from core.config import config

def balance_and_filter_dataset() -> None:
  """
  Refine dataset by filtering in Theta band and balancing trials 1:1.
  Subjects with too few incorrect trials are excluded.
  """
  sub_files: List[Path] = sorted(list(config.paths.EPOCHS_DIR.glob("*_master-epo.fif")))
  
  print("Refining subjects (including all subjects with incorrect trials)...")
  
  valid_count = 0
  for f in sub_files:
    subject_id: str = f.stem.split('_')[0]
    epochs = mne.read_epochs(f, preload=True, verbose=False)
    
    n_correct = len(epochs['Correct'])
    n_incorrect = len(epochs['Incorrect'])
    
    if n_incorrect == 0:
      continue
      
    # 1. Theta Band Filtering (4-8 Hz)
    epochs_theta = epochs.copy().filter(
      l_freq=config.proc.THETA_BAND[0], 
      h_freq=config.proc.THETA_BAND[1], 
      fir_design='firwin', verbose=False
    )
    
    # 2. Trial Balancing (1:1 Ratio)
    n_match = min(n_correct, n_incorrect)
    
    idx_correct = np.random.choice(len(epochs_theta['Correct']), n_match, replace=False)
    idx_incorrect = np.random.choice(len(epochs_theta['Incorrect']), n_match, replace=False)
    
    balanced_epochs = mne.concatenate_epochs([
      epochs_theta['Correct'][idx_correct],
      epochs_theta['Incorrect'][idx_incorrect]
    ], verbose=False)
    
    save_file: Path = config.paths.REFINED_DIR / f"{subject_id}_theta_balanced-epo.fif"
    balanced_epochs.save(save_file, overwrite=True, verbose=False)
    valid_count += 1
    
  print(f"Refinement complete. Total valid subjects processed: {valid_count}")

if __name__ == "__main__":
  balance_and_filter_dataset()
