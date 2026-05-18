import numpy as np
from algorithms.t_algebra import t_prod, t_svd, t_transpose

def test_t_svd_reconstruction():
    n1, n2, n3 = 10, 8, 5
    A = np.random.randn(n1, n2, n3)
    
    U, S, V = t_svd(A)
    
    # Reconstruction: A_hat = U * S * V^T
    Vt = t_transpose(V)
    A_hat = t_prod(t_prod(U, S), Vt)
    
    diff = np.linalg.norm(A - A_hat) / np.linalg.norm(A)
    print(f"Reconstruction relative error: {diff:.2e}")
    assert diff < 1e-10, "t-SVD reconstruction failed!"

if __name__ == "__main__":
    test_t_svd_reconstruction()
    print("t-algebra tests passed!")
