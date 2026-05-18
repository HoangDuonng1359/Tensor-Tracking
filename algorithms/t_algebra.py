import torch
import numpy as np

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def t_prod(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Tensor-tensor product (t-product) using PyTorch FFT.
    A: (n1, n2, n3)
    B: (n2, m2, n3)
    """
    n1, n2, n3 = A.shape
    m1, m2, m3 = B.shape
    
    A_fft = torch.fft.fft(A, dim=2)
    B_fft = torch.fft.fft(B, dim=2)
    
    # Batch matrix multiplication over the 3rd dimension
    # A_fft is (n1, n2, n3), B_fft is (n2, m2, n3)
    # We need (n3, n1, n2) @ (n3, n2, m2)
    C_fft = torch.matmul(A_fft.permute(2, 0, 1), B_fft.permute(2, 0, 1))
    C_fft = C_fft.permute(1, 2, 0) # Back to (n1, m2, n3)
    
    return torch.real(torch.fft.ifft(C_fft, dim=2))

def t_svd(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tensor SVD (t-SVD) using PyTorch.
    """
    n1, n2, n3 = A.shape
    A_fft = torch.fft.fft(A, dim=2)
    
    # PyTorch svd works on batches
    # Input to torch.linalg.svd: (..., M, N)
    U_fft, S_vals, Vh_fft = torch.linalg.svd(A_fft.permute(2, 0, 1), full_matrices=True)
    
    # Construct S_fft as a batch of diagonal matrices
    S_fft = torch.zeros((n3, n1, n2), device=A.device, dtype=A_fft.dtype)
    for i in range(min(n1, n2)):
        S_fft[:, i, i] = S_vals[:, i]
    
    U = torch.real(torch.fft.ifft(U_fft.permute(1, 2, 0), dim=2))
    S = torch.real(torch.fft.ifft(S_fft.permute(1, 2, 0), dim=2))
    V = torch.real(torch.fft.ifft(Vh_fft.permute(1, 2, 0).conj(), dim=2))
    
    return U, S, V

def t_transpose(A: torch.Tensor) -> torch.Tensor:
    """
    Tensor transpose (t-transpose) in PyTorch.
    """
    n1, n2, n3 = A.shape
    # Block circulant structure transpose:
    # First slice is transposed.
    # Other slices are transposed and reversed in order.
    At = torch.zeros((n2, n1, n3), device=A.device, dtype=A.dtype)
    At[:, :, 0] = A[:, :, 0].T
    if n3 > 1:
        At[:, :, 1:] = A[:, :, torch.arange(n3-1, 0, -1)].transpose(0, 1)
    return At
