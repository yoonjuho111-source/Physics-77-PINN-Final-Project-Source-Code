
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# 1. LOAD MASK
mask = loadmat("conductor_data.mat")["mask_crop"].astype(np.float32)
H, W = mask.shape
inside = (mask == 1)

# 2. DATASET CLASS
class SyntheticEITDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        base = self.files[idx]           
        mat_u = loadmat(f"U_sim_{base}.mat")
        mat_s = loadmat(f"sigma_{base}.mat")

        U = mat_u["U_sim"].astype(np.float32)    
        sigma = mat_s["sigma"].astype(np.float32)    

        # Fill NaNs
        U = np.nan_to_num(U, nan=np.nanmean(U[inside]))
        sigma = np.nan_to_num(sigma, nan=np.nanmean(sigma[inside]))

        # Normalize U inside conductor
        mean = U[inside].mean()
        std = U[inside].std() + 1e-8
        U_norm = (U - mean) / std

        # Convert to tensors (1×1×H×W)
        return (
            torch.tensor(U_norm).unsqueeze(0),     # input voltage
            torch.tensor(sigma).unsqueeze(0),      # target conductivity
            torch.tensor(mask).unsqueeze(0)        # mask
        )


# 3. BUILD TRAINING SET

files = sorted(glob.glob("U_sim_*.mat"))

# keep only those ending in a number: U_sim_0.mat, U_sim_1.mat, ...
ids = []
for f in files:
    base = f.split("_")[-1].split(".")[0]
    if base.isdigit():      # only numeric IDs
        ids.append(base)

print("Using synthetic IDs:", ids)

dataset = SyntheticEITDataset(ids)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

print(f"Found {len(dataset)} synthetic samples.")

# 4. CNN ARCHITECTURE
class CNN_Inverse(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.net(x)

model = CNN_Inverse().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# masked loss
def masked_mse(pred, true, mask):
    diff = (pred - true) * mask
    return torch.mean(diff**2)


# 5. TRAINING LOOP
epochs = 2000
print("Training CNN on MULTIPLE samples...")

for epoch in range(epochs):

    epoch_loss = 0

    for U_batch, sigma_batch, mask_batch in loader:
        U_batch = U_batch.to(device)
        sigma_batch = sigma_batch.to(device)
        mask_batch = mask_batch.to(device)

        optimizer.zero_grad()
        pred = model(U_batch)

        loss = masked_mse(pred, sigma_batch, mask_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss = {epoch_loss:.4e}")


# 6. EVALUATE ON ONE SYNTHETIC SAMPLE

print("\nEvaluating on synthetic sample 0...")
sample = dataset[0]
U0, sigma0, mask0 = sample
U0 = U0.unsqueeze(0).to(device)
sigma0 = sigma0.squeeze(0).numpy()

pred0 = model(U0).detach().cpu().numpy()[0,0]

plt.figure(figsize=(14,4))
plt.subplot(1,3,1); plt.title("U_sim_0 (input)"); plt.imshow(sample[0][0], cmap='viridis'); plt.colorbar()
plt.subplot(1,3,2); plt.title("Pred σ"); plt.imshow(pred0, cmap='plasma'); plt.colorbar()
plt.subplot(1,3,3); plt.title("True σ"); plt.imshow(sigma0, cmap='plasma'); plt.colorbar()
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 7. SAVE TRAINED MODEL
# ------------------------------------------------------------
torch.save(model.state_dict(), "cnn_multisample.pt")
print("Saved cnn_multisample.pt")
