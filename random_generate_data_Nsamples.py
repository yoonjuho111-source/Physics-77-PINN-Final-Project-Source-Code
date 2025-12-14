import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter

# PDE SOLVER
def solve_pde(u, sigma, iterations=4000):
    H, W = sigma.shape
    u_old = np.zeros_like(u)
    
    for it in range(iterations):
        u_old[:] = u

        for i in range(1, H-1):
            for j in range(1, W-1):

                if j == 0 or j == W-1:
                    continue

                sxp = 0.5 * (sigma[i,j] + sigma[i,j+1])
                sxm = 0.5 * (sigma[i,j] + sigma[i,j-1])
                syp = 0.5 * (sigma[i,j] + sigma[i+1,j])
                sym = 0.5 * (sigma[i,j] + sigma[i-1,j])

                u[i,j] = (
                    sxp * u[i,j+1] +
                    sxm * u[i,j-1] +
                    syp * u[i+1,j] +
                    sym * u[i-1,j]
                ) / (sxp + sxm + syp + sym + 1e-12)

        # Neumann BC at top/bottom
        u[0,:]  = u[1,:]
        u[-1,:] = u[-2,:]

        # convergence
        if np.max(np.abs(u - u_old)) < 1e-6:
            break

    return u


# Ramdom smooth conductivity
def generate_random_sigma(H, W, mask):
    sigma = np.random.rand(H, W)
    sigma = gaussian_filter(sigma, sigma=5)
    sigma = 0.5 + 1.5 * sigma   
    sigma *= mask         
    return sigma

 
def generate_dataset(N=10, maskfile="conductor_data.mat"):

    mask = sio.loadmat(maskfile)["mask_crop"]
    H, W = mask.shape
    print("Loaded mask shape:", mask.shape)

    for k in range(N):
        print(f"\n===== Generating sample {k+1}/{N} =====")

        sigma = generate_random_sigma(H, W, mask)

        u0 = np.zeros((H, W))
        u0[:, 0]  = 1.0
        u0[:, -1] = 0.0

        U_sim = solve_pde(u0, sigma)

        sio.savemat(f"sigma_{k}.mat", {"sigma": sigma})
        sio.savemat(f"U_sim_{k}.mat", {"U_sim": U_sim})

        print(f"Saved sigma_{k}.mat and U_sim_{k}.mat")

    print("\nFinished generating synthetic dataset!")


# Run generator
generate_dataset(N=10)
