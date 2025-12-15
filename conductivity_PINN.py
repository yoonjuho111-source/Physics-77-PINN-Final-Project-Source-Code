import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import random
import os
import scipy.io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
MODEL_PATH = "pinn_siren_hard_constraint.pth"

# ==========================================
# 🔥 0. Configuration
# ==========================================
FILE_PATH = r'C:\Users\lenovo\Desktop\Berkeley Lilexiao\PINN_Pytorch\conductor_data.mat'
U_NAME = 'U_conductor'
MASK_NAME = 'mask_crop'

# 🎯 Core setting: choose the anchor target value
# No matter whether you set it to 2.0, 4.0, or 8.0, the network will be mathematically forced to match this value.
ANCHOR_TARGET_VAL = 4.0

# Loss weights (Anchor loss no longer needed)
W_DATA = 1000.0
W_PHY = 100.0


# ============================
# 1. Initialization
# ============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


set_seed(42)


# ============================
# 2. Data Loading
# ============================
def load_mat_data(file_path, u_name, mask_name):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    print(f"Loading: {file_path}")
    mat_data = scipy.io.loadmat(file_path)
    if u_name not in mat_data:
        raise KeyError(f"Variable '{u_name}' not found in .mat file")
    u_matrix = mat_data[u_name].astype(np.float32)
    if mask_name not in mat_data:
        raise KeyError(f"Variable '{mask_name}' not found in .mat file")
    mask_matrix = mat_data[mask_name].astype(bool)

    rows, cols = u_matrix.shape
    y = np.linspace(-1, 1, rows)
    x = np.linspace(-1, 1, cols)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.flatten(), Y.flatten()], axis=1)
    U = u_matrix.reshape(-1, 1)
    Mask = mask_matrix.reshape(-1, 1)
    return XY, U, Mask


try:
    XY_data_np, U_data_np, Mask_data_np = load_mat_data(FILE_PATH, U_NAME, MASK_NAME)
except Exception as e:
    print(f"⚠️ Failed to load data ({e}), switching to synthetic data for demonstration...")
    rows, cols = 66, 65
    y = np.linspace(-1, 1, rows)
    x = np.linspace(-1, 1, cols)
    X, Y = np.meshgrid(x, y)
    XY_data_np = np.stack([X.flatten(), Y.flatten()], axis=1)
    # Simulated potential distribution: high in center, low at edges
    U_data_np = (np.exp(-(X ** 2 + Y ** 2)) * 3).reshape(-1, 1)
    Mask_data_np = np.ones_like(U_data_np, dtype=bool)

XY_train = torch.tensor(XY_data_np, dtype=torch.float32).to(device)
XY_train.requires_grad_(True)
U_train_measured = torch.tensor(U_data_np, dtype=torch.float32).to(device)
Mask_train = torch.tensor(Mask_data_np, dtype=torch.bool).to(device)

print(f"✅ Data loaded. Masked points: {Mask_train.sum().item()}")


# ============================
# 3. SIREN Network (with Hard Constraint)
# ============================

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=10):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                k = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-k, k)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-0.0, 0.0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    def __init__(self, in_features=2, hidden_features=128, hidden_layers=4, out_features=1, first_omega_0=10,
                 hidden_omega_0=10):
        super().__init__()
        layers = []
        layers.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            k = np.sqrt(6 / hidden_features) / hidden_omega_0
            final_linear.weight.uniform_(-k, k)
            final_linear.bias.uniform_(-0.0, 0.0)
        layers.append(final_linear)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PhysicsInformedNet(nn.Module):
    def __init__(self, target_val=1.0):
        super(PhysicsInformedNet, self).__init__()
        self.target_val = target_val

        # Define networks
        self.net_u = Siren(in_features=2, hidden_features=128, hidden_layers=4, out_features=1)
        self.net_sigma_raw = Siren(in_features=2, hidden_features=128, hidden_layers=4, out_features=1)

        # Register anchor location (0,0) as buffer (not trainable)
        self.register_buffer('anchor_loc', torch.tensor([[-0.5, 0.0]], dtype=torch.float32))

    def forward(self, x):
        u_pred = self.net_u(x)

        # 1. Compute base sigma (>0)
        sigma_raw = self.net_sigma_raw(x)
        sigma_base = F.softplus(sigma_raw + 1.0)

        # 2. Compute sigma at anchor point
        sigma_anchor_raw = self.net_sigma_raw(self.anchor_loc)
        sigma_anchor_base = F.softplus(sigma_anchor_raw + 1.0)

        # 3. Hard constraint: enforce sigma(anchor) = target_val
        sigma_pred = sigma_base / (sigma_anchor_base + 1e-6) * self.target_val

        return u_pred, sigma_pred


# ============================
# 4. Loss Computation (Anchor loss removed)
# ============================
def calculate_loss_components(model, xy_in, u_measured, mask):
    u_pred, sigma_pred = model(xy_in)

    # --- Data Loss ---
    loss_data = torch.mean((u_pred[mask] - u_measured[mask]) ** 2)

    # --- Physics Loss ---
    grad_u = torch.autograd.grad(u_pred, xy_in, torch.ones_like(u_pred), create_graph=True)[0]
    u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]

    J_x = sigma_pred * u_x
    J_y = sigma_pred * u_y

    grad_Jx = torch.autograd.grad(J_x, xy_in, torch.ones_like(J_x), create_graph=True)[0]
    grad_Jy = torch.autograd.grad(J_y, xy_in, torch.ones_like(J_y), create_graph=True)[0]

    pde_residual = grad_Jx[:, 0:1] + grad_Jy[:, 1:2]

    if mask.sum() > 0:
        loss_phy = torch.mean(pde_residual[mask] ** 2)
    else:
        loss_phy = torch.tensor(0.0, device=device)

    return loss_data, loss_phy


# ============================
# 5. Training
# ============================
NUM_ADAN_EPOCHS = 8000
NUM_LBFGS_STEPS = 5000

# Initialize model with target anchor value
model = PhysicsInformedNet(target_val=ANCHOR_TARGET_VAL).to(device)


def train():
    # Force fresh training
    if os.path.exists(MODEL_PATH):
        try:
            os.remove(MODEL_PATH)
            print("🧹 Old model removed. Starting fresh training.")
        except:
            print("⚠️ Unable to delete old model. Continuing anyway.")

    print(f"🚀 Training started (Hard Constraint Mode) | Target Anchor = {ANCHOR_TARGET_VAL}")

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer_adam, mode='min', factor=0.5, patience=500, min_lr=1e-6)
    pbar_adam = tqdm(range(NUM_ADAN_EPOCHS), desc="Adam")

    for epoch in pbar_adam:
        optimizer_adam.zero_grad()
        loss_data, loss_phy = calculate_loss_components(model, XY_train, U_train_measured, Mask_train)

        loss_total = (W_DATA * loss_data) + (W_PHY * loss_phy)

        loss_total.backward()
        optimizer_adam.step()
        scheduler.step(loss_total.detach())

        if epoch % 100 == 0:
            pbar_adam.set_postfix({
                'Total': f'{loss_total.item():.2e}',
                'Data': f'{loss_data.item():.2e}',
                'Phy': f'{loss_phy.item():.2e}'
            })

    print("\nSwitching to L-BFGS")
    optimizer_LBGFS = torch.optim.LBFGS(
        model.parameters(), lr=0.5, max_iter=NUM_LBFGS_STEPS,
        history_size=50, tolerance_change=1.0 * np.finfo(float).eps,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer_LBGFS.zero_grad()
        loss_data, loss_phy = calculate_loss_components(model, XY_train, U_train_measured, Mask_train)
        total_loss = (W_DATA * loss_data) + (W_PHY * loss_phy)
        total_loss.backward()

        if closure.step % 200 == 0:
            print(f"L-BFGS Step {closure.step}: Total={total_loss.item():.5f}")
        closure.step += 1
        return total_loss.detach()

    closure.step = 0
    optimizer_LBGFS.step(closure)
    torch.save({'model_state_dict': model.state_dict()}, MODEL_PATH)
    print("Training completed and model saved.")


train()

# ============================
# 6. Visualization & Validation
# ============================
model.eval()
with torch.no_grad():
    u_pred, sigma_pred = model(XY_train)

    # Validate hard constraint
    anchor_input = torch.tensor([[0.0, 0.0]], device=device)
    _, s_anchor = model(anchor_input)
    print("=" * 40)
    print(f"🎯 Anchor Target : {ANCHOR_TARGET_VAL}")
    print(f"🤖 Model Output at (0,0): {s_anchor.item():.6f}")
    print("=" * 40)

    total = XY_train.shape[0]
    if total == 4290:
        rows, cols = 66, 65
    else:
        import math
        rows = int(math.sqrt(total))
        cols = total // rows

    U_grid_pred = u_pred.cpu().numpy().reshape(rows, cols)
    Sigma_grid_pred = sigma_pred.cpu().numpy().reshape(rows, cols)
    U_grid_true = U_train_measured.cpu().numpy().reshape(rows, cols)
    Mask_grid = Mask_train.cpu().numpy().reshape(rows, cols)

    U_grid_pred[~Mask_grid] = np.nan
    Sigma_grid_pred[~Mask_grid] = np.nan
    U_grid_true[~Mask_grid] = np.nan

    vmin_u = np.nanmin(U_grid_true)
    vmax_u = np.nanmax(U_grid_true)
    extent_range = [-1, 1, -1, 1]

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.title("Measured U")
    plt.imshow(U_grid_true, origin='lower', cmap='RdBu', aspect='auto', vmin=vmin_u, vmax=vmax_u, extent=extent_range)
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Predicted U")
    plt.imshow(U_grid_pred, origin='lower', cmap='RdBu', aspect='auto', vmin=vmin_u, vmax=vmax_u, extent=extent_range)
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title(f"Predicted Sigma (Fixed @ {ANCHOR_TARGET_VAL})")
    plt.imshow(Sigma_grid_pred, origin='lower', cmap='viridis', aspect='auto', extent=extent_range)
    plt.colorbar()

    plt.scatter([0], [0], c='red', marker='x', s=100, label='Anchor')
    plt.legend()

    plt.tight_layout()
    plt.show()
