import os
import copy
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error

COLUMN_NAMES = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)

SELECTED_SENSORS = [
    "s2", "s3", "s4", "s6", "s7", "s8",
    "s9", "s11", "s12", "s13", "s14",
    "s15", "s17", "s20", "s21",
]
 
MAX_RUL     = 125
SEQ_LEN     = 30
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.2

class TemporalAttention(nn.Module):
    """Soft attention over the time dimension."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
 
    def forward(self, x):            # x: (B, T, H)
        scores  = self.attn(x)       # (B, T, 1)
        weights = torch.softmax(scores, dim=1)    # (B, T, 1)
        context = (weights * x).sum(dim=1)        # (B, H)
        return context, weights.squeeze(-1)
 
 
class GRULSTM(nn.Module):
    """
    GRU block → LSTM block → Temporal Attention → FC head
    Copied verbatim from GRU_LSTM.ipynb, Section 5.
    """
    def __init__(self,
                 input_size:  int   = 18,
                 gru_hidden:  int   = 64,
                 gru_layers:  int   = 2,
                 lstm_hidden: int   = 64,
                 lstm_layers: int   = 2,
                 fc_hidden:   int   = 32,
                 dropout:     float = 0.3,
                 **kwargs):          # absorb extra checkpoint keys safely
        super().__init__()
 
        # GRU block
        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = gru_hidden,
            num_layers  = gru_layers,
            batch_first = True,
            dropout     = dropout if gru_layers > 1 else 0.0,
            bidirectional = False
        )
        self.gru_drop = nn.Dropout(dropout)
 
        # LSTM block
        self.lstm = nn.LSTM(
            input_size  = gru_hidden,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0,
            bidirectional = False
        )
        self.lstm_drop = nn.Dropout(dropout)
 
        # Temporal attention
        self.attention = TemporalAttention(lstm_hidden)
 
        # Regression head
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
            nn.Sigmoid()    # output in [0,1]; scale back by max_rul
        )
 
    def forward(self, x, return_attn=False):
        gru_out,  _     = self.gru(x)            # (B, T, GRU_H)
        gru_out         = self.gru_drop(gru_out)
        lstm_out, _     = self.lstm(gru_out)     # (B, T, LSTM_H)
        lstm_out        = self.lstm_drop(lstm_out)
        context, attn_w = self.attention(lstm_out)
        out             = self.fc(context)        # (B, 1)
        if return_attn:
            return out, attn_w
        return out
    
# ─────────────────────────────────────────────────────────────────────────────
# 2.  Quantization helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def quantize_model(model: nn.Module) -> nn.Module:
    """
    Dynamic PTQ — targets GRU, LSTM, and all Linear layers.
    Weights → INT8 ahead of time.
    Activations → INT8 on-the-fly during inference.
    """
    m = copy.deepcopy(model).cpu().eval()
    return torch.quantization.quantize_dynamic(
        m,
        qconfig_spec={nn.GRU, nn.LSTM, nn.Linear},   # ← GRU added
        dtype=torch.qint8
    )
 
 
def get_size_mb(model: nn.Module, tmp: str = "_tmp_ckpt.pt") -> float:
    torch.save(model.state_dict(), tmp)
    mb = os.path.getsize(tmp) / (1024 ** 2)
    os.remove(tmp)
    return mb
 
 
def benchmark_ms(model: nn.Module, X: np.ndarray,
                 batch_size: int = 64, runs: int = 5) -> float:
    model.eval()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X)),
        batch_size=batch_size
    )
    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            for (b,) in loader:
                model(b.cpu())
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))
 
 
def predict(model: nn.Module, X: np.ndarray,
            batch_size: int = 64) -> np.ndarray:
    model.eval()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X)),
        batch_size=batch_size
    )
    preds = []
    with torch.no_grad():
        for (b,) in loader:
            preds.append(model(b.cpu()).numpy())
    return np.concatenate(preds).flatten()
 
 
def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0,
                                 np.exp(-d / 13) - 1,
                                 np.exp(d / 10) - 1)))

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Main
# ─────────────────────────────────────────────────────────────────────────────
 
def main(args):
    print(f"\n{'='*58}")
    print(f"  GRU-LSTM Dynamic Quantization  |  Dataset: {args.dataset}")
    print(f"{'='*58}\n")
 
    # ── Load checkpoint ──────────────────────────────────────────────────────
    model_path = args.model_path or f"gru_lstm_{args.dataset}.pt"
    print(f"Loading checkpoint: {model_path}")
 
    ck      = torch.load(model_path, map_location="cpu")
    config  = ck["config"]
    max_rul = config["max_rul"]
 
    print(f"  Config  : {config}")
 
    # ── Rebuild model from saved config ──────────────────────────────────────
    fp32_model = GRULSTM(**config).cpu().eval()
    fp32_model.load_state_dict(ck["model_state"])
    print(f"  Params  : {sum(p.numel() for p in fp32_model.parameters()):,}\n")
 
    # ── Load test tensors from .npz ──────────────────────────────────────────
    npz_path = args.npz_path or f"{args.dataset}_tensors.npz"
    print(f"Loading test data : {npz_path}")
 
    npz       = np.load(npz_path)
    X_test    = npz["X_test"].astype(np.float32)   # (E, 30, 18)
    y_test    = npz["y_test"].astype(np.float32)   # (E,)  — already in cycles
 
    # If labels were saved normalised [0,1], scale back
    if y_test.max() <= 1.0:
        y_test = y_test * max_rul
 
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_test  : min={y_test.min():.0f}  max={y_test.max():.0f}\n")
 
    # ── FP32 baseline ────────────────────────────────────────────────────────
    print("Running FP32 baseline ...")
    fp32_size  = get_size_mb(fp32_model)
    fp32_time  = benchmark_ms(fp32_model, X_test)
    fp32_raw   = predict(fp32_model, X_test)        # in [0,1]
    fp32_preds = fp32_raw * max_rul                 # scale to cycles
    fp32_rmse  = np.sqrt(mean_squared_error(y_test, fp32_preds))
    fp32_mae   = mean_absolute_error(y_test, fp32_preds)
    fp32_nasa  = nasa_score(y_test, fp32_preds)
 
    # ── INT8 quantized ───────────────────────────────────────────────────────
    print("Applying dynamic quantization (GRU + LSTM + Linear → INT8) ...")
    int8_model = quantize_model(fp32_model)
    int8_size  = get_size_mb(int8_model)
    int8_time  = benchmark_ms(int8_model, X_test)
    int8_raw   = predict(int8_model, X_test)
    int8_preds = int8_raw * max_rul
    int8_rmse  = np.sqrt(mean_squared_error(y_test, int8_preds))
    int8_mae   = mean_absolute_error(y_test, int8_preds)
    int8_nasa  = nasa_score(y_test, int8_preds)
 
    # ── Deltas ───────────────────────────────────────────────────────────────
    size_reduction = (1 - int8_size / fp32_size) * 100
    speedup        = fp32_time / int8_time
    rmse_delta     = (int8_rmse - fp32_rmse) / fp32_rmse * 100
    mae_delta      = (int8_mae  - fp32_mae)  / fp32_mae  * 100
    nasa_delta     = (int8_nasa - fp32_nasa) / abs(fp32_nasa) * 100
 
    # ── Report ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*58}")
    print(f"  {'Metric':<24} {'FP32':>10} {'INT8':>10} {'Δ':>10}")
    print(f"{'─'*58}")
    print(f"  {'Model size (MB)':<24} {fp32_size:>10.3f} {int8_size:>10.3f} {-size_reduction:>+9.1f}%")
    print(f"  {'Inference (ms)':<24} {fp32_time:>10.1f} {int8_time:>10.1f} {speedup:>9.2f}x")
    print(f"  {'RMSE (cycles)':<24} {fp32_rmse:>10.4f} {int8_rmse:>10.4f} {rmse_delta:>+9.2f}%")
    print(f"  {'MAE  (cycles)':<24} {fp32_mae:>10.4f}  {int8_mae:>10.4f} {mae_delta:>+9.2f}%")
    print(f"  {'NASA Score':<24} {fp32_nasa:>10.1f} {int8_nasa:>10.1f} {nasa_delta:>+9.2f}%")
    print(f"{'─'*58}")
 
    # ── Verdict ──────────────────────────────────────────────────────────────
    THRESHOLD = 5.0   # % RMSE degradation considered acceptable
 
    if rmse_delta <= THRESHOLD:
        print(f"\n  ✅  PTQ SUCCESSFUL — RMSE degraded by only {rmse_delta:+.2f}%")
        save_path = f"gru_lstm_int8_{args.dataset}.pt"
        # Save full model object (quantized models can't use just state_dict)
        torch.save(int8_model, save_path)
        print(f"  INT8 model saved → {save_path}")
        print(f"\n  Reload with:")
        print(f"    model = torch.load('{save_path}', map_location='cpu')")
        print(f"    preds = model(X_new) * {max_rul}")
    else:
        print(f"\n  ⚠️   PTQ FAILED — RMSE degraded by {rmse_delta:+.2f}% (threshold: {THRESHOLD}%)")
        print(f"  Upgrade to Quantization-Aware Training (QAT):\n")
        print(f"    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')")
        print(f"    torch.quantization.prepare_qat(model, inplace=True)")
        print(f"    # retrain for ~10% of original epochs (~6 epochs here)")
        print(f"    torch.quantization.convert(model.eval(), inplace=True)")
 
    print(f"\n{'='*58}\n")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    type=str, default="FD001",
                        choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to .pt checkpoint (default: gru_lstm_<dataset>.pt)")
    parser.add_argument("--npz_path",   type=str, default=None,
                        help="Path to _tensors.npz (default: <dataset>_tensors.npz)")
    args = parser.parse_args()
    main(args)