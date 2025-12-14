import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running trained CNN on REAL voltage. Device =", device)

# CNN 
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

# load saved CNN weights from synthetic data 
model = CNN_Inverse().to(device)
model.load_state_dict(torch.load("cnn_multisample.pt", map_location=device))
model.eval()
print("Loaded cnn_multisample.pt successfully!")

# Real experimental voltage with mask 
data = loadmat("conductor_data.mat")
U_real = data["U_conductor"] # shape is (66,65)
mask   = data["mask_crop"]

U_real = np.nan_to_num(U_real, nan=np.nanmean(U_real))
print("Loaded U_real shape:", U_real.shape)

# normalization of real voltage 
U_mean = U_real[mask == 1].mean()
U_std  = U_real[mask == 1].std() + 1e-8

U_real_norm = (U_real - U_mean) / U_std

# running prediction
U_input = torch.tensor(U_real_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    sigma_pred_real = model(U_input).cpu().numpy()[0, 0]

#FInal save 
savemat("sigma_pred_real.mat", {"sigma_pred_real": sigma_pred_real})
print("Saved sigma_pred_real.mat")

# visualize

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.title("Real Voltage U_real (normalized)")
plt.imshow(U_real_norm, cmap="bwr")
plt.colorbar()

plt.subplot(1,2,2)
plt.title("CNN Predicted σ(x,y) on REAL voltage")
plt.imshow(sigma_pred_real, cmap = "bwr")
plt.colorbar()

plt.tight_layout()
plt.show()

