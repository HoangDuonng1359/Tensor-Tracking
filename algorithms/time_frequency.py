import torch
import numpy as np
from typing import Optional

class RIDRihaczek:
  """
  GPU-accelerated RID-Rihaczek Time-Frequency Distribution (TFD) engine.
  Implements Formula (5) from Ozdemir (2017) with Choi-Williams exponential kernel.
  """
  def __init__(self, n_times: int, sigma: float = 1.0):
    self.n_times = n_times
    self.sigma = sigma
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Precompute kernel in ambiguity domain (theta-tau)
    # theta: doppler, tau: lag
    theta = torch.fft.fftfreq(n_times, d=1.0, device=self.device) * 2 * np.pi
    tau = torch.arange(-(n_times // 2), n_times - (n_times // 2), device=self.device).float()
    THETA, TAU = torch.meshgrid(theta, tau, indexing='ij')
    
    # g(theta, tau) = exp(-(theta*tau)^2 / sigma) * exp(j * theta * tau / 2)
    # The paper uses exp(j * theta * tau / 2) as the Rihaczek component
    self.kernel = torch.exp(-(THETA * TAU)**2 / self.sigma).to(torch.complex64) * torch.exp(1j * THETA * TAU / 2.0)

  def compute_tfd(self, signal_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the complex TFD for a batch of signals.
    signal_tensor: (batch_size, n_times)
    returns: (batch_size, n_times, n_times) in (time, frequency)
    """
    batch_size, n_times = signal_tensor.shape
    device = signal_tensor.device
    
    # 1. Compute Analytic Signal (Hilbert Transform)
    f = torch.fft.fft(signal_tensor, dim=1)
    f[:, 1:(n_times+1)//2] *= 2
    f[:, (n_times+1)//2:] = 0
    analytic = torch.fft.ifft(f, dim=1)
    
    # 2. Compute Local Autocorrelation Function (AF) in (time, lag) domain
    # R(t, tau) = x(t) * conj(x(t-tau))
    R = torch.zeros((batch_size, n_times, n_times), dtype=torch.complex64, device=device)
    for tau_idx, tau in enumerate(range(-(n_times // 2), n_times - (n_times // 2))):
      shifted = torch.roll(analytic, shifts=-tau, dims=1)
      R[:, :, tau_idx] = analytic * torch.conj(shifted)
      
    # 3. Transform to Ambiguity Domain (theta, tau)
    # A(theta, tau) = FFT_t [ R(t, tau) ]
    A = torch.fft.fft(R, dim=1)
    
    # 4. Apply RID-Rihaczek Kernel
    # C_A(theta, tau) = A(theta, tau) * g(theta, tau)
    A_filtered = A * self.kernel.unsqueeze(0)
    
    # 5. Transform back to TFR Domain (t, omega)
    # C(t, omega) = IFFT_theta [ FFT_tau [ A_filtered(theta, tau) ] ]
    # Note: Rihaczek is defined with specific FFT orientations
    tfr = torch.fft.ifft(torch.fft.ifft(A_filtered, dim=2), dim=1)
    
    return tfr

  def compute_phase(self, signal_tensor: torch.Tensor) -> torch.Tensor:
    """
    Estimate phase in the frequency domain after TFD.
    Returns the angle of the complex TFD.
    """
    tfr = self.compute_tfd(signal_tensor)
    return torch.angle(tfr)
