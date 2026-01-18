
import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ap = argparse.ArgumentParser()
ap.add_argument("--save-dir", type=str, default="sedac_data")
ap.add_argument("--hidden-path", type=str, default="")
ap.add_argument("--entropy-path", type=str, default="")
ap.add_argument("--out-path", type=str, default="")
ap.add_argument("--probe-rank", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--batch-size", type=int, default=1024)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--grad-clip", type=float, default=1.0)
args = ap.parse_args()

SAVE_DIR = str(args.save_dir)
HIDDEN_PATH = str(args.hidden_path) if str(args.hidden_path) else os.path.join(SAVE_DIR, "hidden_states_layer21.pt")
ENTROPY_PATH = str(args.entropy_path) if str(args.entropy_path) else os.path.join(SAVE_DIR, "entropies_layer21.pt")
MODEL_PATH = str(args.out_path) if str(args.out_path) else os.path.join(SAVE_DIR, "sedac_probe_layer21.pth")

PROBE_RANK = int(args.probe_rank)
LEARNING_RATE = float(args.lr)
BATCH_SIZE = int(args.batch_size)
EPOCHS = int(args.epochs)

# --- 1. 定义探针模型 (LRE Probe) ---
class LREProbe(nn.Module):
    def __init__(self, input_dim, rank=64):
        super().__init__()
        # 结构：Linear(d -> r) -> LayerNorm -> Linear(r -> 1) -> Softplus
        self.proj = nn.Linear(input_dim, rank, bias=False)
        self.norm = nn.LayerNorm(rank)
        self.head = nn.Linear(rank, 1)
        self.act = nn.Softplus() # 保证输出熵非负

    def forward(self, x):
        # x: [batch, input_dim]
        # x 可能包含 nan/inf，先处理一下
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = self.proj(x)
        h = self.norm(h)
        return self.act(self.head(h))

# --- 2. 加载数据 ---
torch.manual_seed(int(args.seed))

print("Loading data...")
hidden_states = torch.load(HIDDEN_PATH, map_location="cpu")
entropies = torch.load(ENTROPY_PATH, map_location="cpu")

# 替换掉数据里的 nan
hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=0.0, neginf=0.0)
entropies = torch.nan_to_num(entropies, nan=0.0, posinf=0.0, neginf=0.0)

# 强制转换类型，确保匹配
hidden_states = hidden_states.float()
entropies = entropies.float()

if hidden_states.ndim != 2:
    raise RuntimeError(f"hidden_states must be [N, H], got {tuple(hidden_states.shape)}")
if entropies.ndim != 1:
    raise RuntimeError(f"entropies must be [N], got {tuple(entropies.shape)}")
if hidden_states.shape[0] != entropies.shape[0]:
    raise RuntimeError(f"N mismatch hidden_states={hidden_states.shape[0]} entropies={entropies.shape[0]}")

HIDDEN_DIM = int(hidden_states.shape[1])
print(f"Data loaded. Samples: {hidden_states.shape[0]} HiddenDim: {HIDDEN_DIM}")
if hidden_states.shape[0] == 0:
    print("Error: No valid samples found!")
    exit(1)

entropies = torch.log1p(entropies)
q_low = float(torch.quantile(entropies, 0.01).item())
q_high = float(torch.quantile(entropies, 0.99).item())
entropies = torch.clamp(entropies, min=q_low, max=q_high)
mu = float(entropies.mean().item())
sigma = float(entropies.std(unbiased=False).item())
sigma = sigma if sigma > 1e-6 else 1e-6
entropies = (entropies - mu) / sigma

dataset = TensorDataset(hidden_states, entropies)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# --- 3. 训练 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {device}...")

probe = LREProbe(HIDDEN_DIM, PROBE_RANK).to(device)
optimizer = optim.AdamW(probe.parameters(), lr=LEARNING_RATE, eps=1e-8)
criterion_huber = nn.SmoothL1Loss(beta=0.5)
criterion_mse = nn.MSELoss()

best_loss = float('inf')
loss_history = []

for epoch in range(EPOCHS):
    probe.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device).unsqueeze(1) # y shape: [batch, 1]
        
        optimizer.zero_grad()
        pred = probe(x)
        loss = (criterion_huber(pred, y) if epoch < max(1, EPOCHS // 2) else criterion_mse(pred, y))
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), float(args.grad_clip))
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
    
    avg_loss = total_loss / train_size
    loss_history.append(avg_loss)
    
    # 验证
    probe.eval()
    val_loss = 0
    val_mse = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            pred = probe(x)
            loss = criterion_mse(pred, y)
            val_loss += loss.item() * x.size(0)
            val_mse += loss.item() * x.size(0)
    avg_val_loss = val_loss / test_size
    avg_val_mse = val_mse / test_size
    
    print(f"Epoch {epoch+1}/{EPOCHS} | TrainLoss: {avg_loss:.6f} | ValMSE: {avg_val_mse:.6f}")
    
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(probe.state_dict(), MODEL_PATH)
        print("  -> Model saved!")

print(f"✅ Training done! Best Val Loss: {best_loss:.6f}")
print(f"Probe saved to: {MODEL_PATH}")

# 简单的可视化（如果可能）
try:
    plt.plot(loss_history)
    plt.title("Probe Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.savefig(os.path.join(SAVE_DIR, "training_curve.png"))
    print("Loss curve saved.")
except:
    pass
