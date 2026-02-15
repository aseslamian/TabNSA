# ===================== GEMMA + TabNSA: E2E METRICS + FLOPs/MACs =====================
import os, json, time, platform, psutil, datetime, glob, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from contextlib import contextmanager
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from thop import profile, clever_format

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5            # start small, scale later
BATCH_SIZE = 32
MAX_LEN = 256         # prompts are short; helps memory & speed
MEASURE_LLM_FLOPS = False   # True = try FLOPs for the full Gemma pass (can be slow/heavy)

# ------------------------------- Security: HF token -------------------------------
tok = os.getenv("Enter your Token")
if tok: login(tok)

# -------------------------- E2E timing / hardware logger --------------------------
def _gpu_info():
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        return {
            "name": torch.cuda.get_device_name(dev),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(dev))),
            "driver": torch.version.cuda,
        }
    return None

def _sys_spec():
    vm = psutil.virtual_memory()
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_gb": round(vm.total / (1024**3), 2),
        "gpu": _gpu_info(),
    }

@contextmanager
def stage(timer_dict, name):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timer_dict[name] = timer_dict.get(name, 0.0) + (time.perf_counter() - t0)

def run_with_metrics(run_name, fn_preprocess, fn_dataload, fn_train, fn_eval, out_dir="logs"):
    os.makedirs(out_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    timers, t0 = {}, time.perf_counter()
    with stage(timers, "preprocess"):   fn_preprocess()
    with stage(timers, "data_loading"): fn_dataload()
    with stage(timers, "train"):        fn_train()
    with stage(timers, "eval"):         metrics = fn_eval()

    total_time = time.perf_counter() - t0
    gpu_peak_gb = (torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0
    cpu_rss_gb = psutil.Process(os.getpid()).memory_info().rss / (1024**3)

    payload = {
        "run_name": run_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "hardware": _sys_spec(),
        "timers_sec": {k: round(v, 4) for k, v in timers.items()},
        "total_time_min": round(total_time / 60.0, 3),
        "gpu_peak_gb": round(gpu_peak_gb, 3),
        "cpu_peak_rss_gb": round(cpu_rss_gb, 3),
        "metrics": metrics,
    }
    path = os.path.join(out_dir, f"{run_name.replace(' ', '_')}.json")
    with open(path, "w") as f: json.dump(payload, f, indent=2)
    print(f"[{run_name}] Runtime: {payload['total_time_min']:.2f} min | "
          f"GPU: {payload['gpu_peak_gb']:.2f} GB | CPU: {payload['cpu_peak_rss_gb']:.2f} GB | "
          f"timers: {payload['timers_sec']} | metrics: {payload['metrics']} | log: {path}")
    return payload

# --------------------------------- Data utilities --------------------------------
def load_csv_folder(folder_glob):
    files = glob.glob(folder_glob)
    if not files:
        raise FileNotFoundError(f"No CSV files found for pattern: {folder_glob}")
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)

def row_to_prompt(row, feature_cols):
    return ", ".join([f"{col} is {row[col]}" for col in feature_cols])

def make_prompt_df(df):
    feature_cols = df.columns[:-1].tolist()
    label_col = df.columns[-1]
    out = pd.DataFrame({
        "prompt": [row_to_prompt(r, feature_cols) for _, r in df.iterrows()],
        "label": df[label_col].values
    })
    return out

# ------------------------------- Torch dataset class ------------------------------
class PromptDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }

# --------------------------------- TabNSA module ---------------------------------
class TabNSA(nn.Module):
    def __init__(self, input_shape, output_shape,
                 dim_head=24, heads=1, sliding_window_size=1,
                 compress_block_size=12, selection_block_size=16, num_selected_blocks=4):
        super().__init__()
        from native_sparse_attention_pytorch import SparseAttention
        from tabmixer import TabMixer
        self.dim = 64
        self.feature_embedding = nn.Linear(1, self.dim)
        self.attention = SparseAttention(
            dim=self.dim,
            dim_head=dim_head,
            heads=heads,
            sliding_window_size=sliding_window_size,
            compress_block_size=compress_block_size,
            selection_block_size=selection_block_size,
            num_selected_blocks=num_selected_blocks
        )
        self.tabmixer = TabMixer(
            dim_tokens=input_shape,
            dim_features=self.dim,
            dim_feedforward=256
        )
        self.head = nn.Sequential(
            nn.Linear(self.dim, 32),
            nn.GELU(),
            nn.Linear(32, output_shape)
        )

    def forward(self, x):     # x: [B, F] where F == input_shape (NUM_FEATURES)
        x = x.unsqueeze(-1)   # [B, F, 1]
        x = self.feature_embedding(x)         # [B, F, 64]
        x_1 = self.attention(x)               # [B, F, 64]
        x_2 = self.tabmixer(x)                # [B, F, 64]
        x = x_1 + x_2
        x = x.mean(dim=1)                     # [B, 64]
        return self.head(x)                   # [B, C]

# ------------------------------ Gemma + adapter + TabNSA -------------------------
class GemmaTabNSA(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size, llm_model, **tabnsa_params):
        super().__init__()
        self.encoder = llm_model

        # (A) Freeze all, then unfreeze last 2 decoder blocks for fine-tuning (Gemma uses model.layers.N)
        for p in self.encoder.parameters(): p.requires_grad = False
        for name, p in self.encoder.named_parameters():
            if name.startswith("model.layers.") and any(name.startswith(f"model.layers.{k}.") for k in [-2, -1]):
                p.requires_grad = True

        # (B) Alternative: LoRA (recommended for speed/memory). Uncomment to use instead of (A).
        # from peft import get_peft_model, LoraConfig, TaskType
        # lora_cfg = LoraConfig(
        #     r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        #     task_type=TaskType.CAUSAL_LM,
        #     target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        # )
        # self.encoder = get_peft_model(self.encoder, lora_cfg)

        self.adapter = nn.Linear(hidden_size, num_features)
        self.tabnsa = TabNSA(input_shape=num_features, output_shape=num_classes, **tabnsa_params)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = out.hidden_states[-1]                      # [B, T, H]
        mask = attention_mask.bool()                        # [B, T]
        hidden_masked = hidden.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        pooled_max, _ = hidden_masked.max(dim=1)            # [B, H]
        pooled_mean = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        adapted = self.adapter(pooled_max) + pooled_mean    # [B, H] -> adapter -> [B, F] plus residual
        return self.tabnsa(adapted)                         # logits [B, C]

# -------------------------------- Train / eval loops -----------------------------
def fit(model, criterion, optimizer, loader, device, epochs=3, val_loader=None, patience=5):
    history, best_loss, bad = {"train_loss": [], "val_loss": []}, float("inf"), 0
    model.to(device)
    for ep in range(epochs):
        model.train(); run = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            run += loss.item() * input_ids.size(0)
        tr_loss = run / len(loader.dataset); history["train_loss"].append(tr_loss)
        if val_loader is not None:
            model.eval(); vrun = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    ids = batch["input_ids"].to(device)
                    ms  = batch["attention_mask"].to(device)
                    y   = batch["labels"].to(device)
                    vloss = criterion(model(ids, ms), y)
                    vrun += vloss.item() * ids.size(0)
            vloss = vrun / len(val_loader.dataset); history["val_loss"].append(vloss)
            if vloss < best_loss:
                best_loss, bad, best_state = vloss, 0, {k: v.cpu() for k,v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    if best_loss < float("inf"): model.load_state_dict(best_state)
                    break
    return history

@torch.no_grad()
def evaluate_model(model, loader, num_classes, device):
    model.eval()
    preds, probs, labels = [], [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        ms  = batch["attention_mask"].to(device)
        y   = batch["labels"].to(device)
        p = torch.softmax(model(ids, ms), dim=1)
        preds.append(p.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
        labels.append(y.cpu().numpy())
    preds = np.concatenate(preds); probs = np.concatenate(probs); labels = np.concatenate(labels)
    from sklearn.metrics import accuracy_score, roc_auc_score
    acc = float(accuracy_score(labels, preds))
    if num_classes == 2:
        auc = float(roc_auc_score(labels, probs[:,1]))
    else:
        auc = float(roc_auc_score(labels, probs, multi_class="ovo"))
    return {"acc": acc, "auc": auc}

# ------------------------------ FLOPs / MACs helpers -----------------------------
def measure_adapter_tabnsa_flops(adapter, tabnsa, hidden_size, num_features, device='cpu'):
    """Measure FLOPs/MACs for adapter + TabNSA (fast & comparable)."""
    class AdapterTab(nn.Module):
        def __init__(self, a, t): super().__init__(); self.a=a; self.t=t
        def forward(self, h): return self.t(self.a(h))
    head = AdapterTab(adapter, tabnsa).to(device).eval()
    dummy = torch.randn(1, hidden_size, device=device)  # pooled LLM embedding per sample
    macs, params = profile(head, inputs=(dummy,), verbose=False)
    flops = 2 * macs
    macs_h, params_h = clever_format([macs, params], "%.3f")
    flops_h, _ = clever_format([flops, params], "%.3f")
    return {"macs": macs, "flops": flops, "params": params,
            "macs_human": macs_h, "flops_human": flops_h, "params_human": params_h}

def measure_full_llm_pass_flops(model, seq_len, device='cpu'):
    """Optional: try FLOPs for full Gemma forward (can be slow/huge)."""
    model = model.to(device).eval()
    dummy_ids = torch.ones(1, seq_len, dtype=torch.long, device=device)
    dummy_mask = torch.ones_like(dummy_ids)
    # Wrap to match signature thop expects
    class Wrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m=m
        def forward(self, x):
            return self.m(input_ids=x, attention_mask=torch.ones_like(x), output_hidden_states=True).logits
    w = Wrapper(model)
    macs, params = profile(w, inputs=(dummy_ids,), verbose=False)
    flops = 2 * macs
    macs_h, params_h = clever_format([macs, params], "%.3f")
    flops_h, _ = clever_format([flops, params], "%.3f")
    return {"macs": macs, "flops": flops, "params": params,
            "macs_human": macs_h, "flops_human": flops_h, "params_human": params_h}

# ----------------------------------- Context bag ---------------------------------
ctx = {}
DATASETS = ["Blood"]             # adjust as needed
DATA_ROOT = "/ocean/projects/cis240149p/eslamian/Data"   # change if needed

# ------------------------------------ Stages ------------------------------------
def pp_gemma_tabnsa():
    # load + concatenate CSVs for first dataset in list
    ds = DATASETS[0]
    df = load_csv_folder(os.path.join(DATA_ROOT, ds, "*.csv"))
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split = 128   # few-shot train size
    train_df = df.iloc[:split].copy()
    test_df  = df.iloc[split:].copy()

    tr_prompt = make_prompt_df(train_df)
    te_prompt = make_prompt_df(test_df)

    le = LabelEncoder().fit(tr_prompt["label"])
    tr_prompt["label_id"] = le.transform(tr_prompt["label"])
    te_prompt["label_id"] = le.transform(te_prompt["label"])
    num_classes = len(le.classes_)

    ctx.update(dict(tr_prompt=tr_prompt, te_prompt=te_prompt, num_classes=num_classes, label_encoder=le))

def dl_gemma_tabnsa():
    # tokenizer + encodings
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it")
    enc_tr = tokenizer(ctx["tr_prompt"]["prompt"].tolist(), padding="max_length",
                       truncation=True, max_length=MAX_LEN, return_tensors="pt")
    enc_te = tokenizer(ctx["te_prompt"]["prompt"].tolist(), padding="max_length",
                       truncation=True, max_length=MAX_LEN, return_tensors="pt")

    ds_tr = PromptDataset(enc_tr, ctx["tr_prompt"]["label_id"].tolist())
    ds_te = PromptDataset(enc_te, ctx["te_prompt"]["label_id"].tolist())

    # build a small val split from train
    n = len(ds_tr); n_val = max(1, int(0.1*n)); n_tr = n - n_val
    tr_set, va_set = torch.utils.data.random_split(ds_tr, [n_tr, n_val], generator=torch.Generator().manual_seed(42))

    ctx["train_loader"] = DataLoader(tr_set, batch_size=BATCH_SIZE, shuffle=True)
    ctx["val_loader"]   = DataLoader(va_set, batch_size=BATCH_SIZE)
    ctx["test_loader"]  = DataLoader(ds_te, batch_size=BATCH_SIZE)
    
    llm = AutoModelForCausalLM.from_pretrained(
    "google/gemma-1.1-2b-it",
    device_map="auto",
    torch_dtype=torch.float32,   
    )   
    
    hidden_size = llm.config.hidden_size
    num_features = hidden_size 

    # best params from your snippet (minus lr/batch handled outside)
    tab_kwargs = dict(dim_head=24, heads=1, sliding_window_size=1,
                      compress_block_size=12, selection_block_size=16, num_selected_blocks=4)

    model = GemmaTabNSA(num_features=num_features,
                        num_classes=ctx["num_classes"],
                        hidden_size=hidden_size,
                        llm_model=llm,
                        **tab_kwargs).to(device).to(torch.float32)

    ctx.update(dict(model=model, hidden_size=hidden_size, num_features=num_features))

def tr_gemma_tabnsa():
    model = ctx["model"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-5)
    criterion = nn.CrossEntropyLoss()
    fit(model, criterion, optimizer, ctx["train_loader"], device=device, epochs=EPOCHS, val_loader=ctx["val_loader"], patience=3)

    # --- params count (trainable & total) ---
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ctx["param_counts"] = {"trainable": int(trainable), "total": int(total),
                           "pct": round(100.0 * trainable / total, 4)}

    # --- FLOPs/MACs for adapter+TabNSA only (fast, comparable) ---
    base = model  # not DataParallel here; if DP, unwrap .module
    flops_head = measure_adapter_tabnsa_flops(base.adapter, base.tabnsa, ctx["hidden_size"], ctx["num_features"],
                                              device=device if device.type=="cuda" else "cpu")
    ctx["flops_head"] = flops_head
    print(f"[Adapter+TabNSA] {flops_head['flops_human']} FLOPs | {flops_head['macs_human']} MACs | Params: {flops_head['params_human']}")

    # --- Optional: FLOPs for the full LLM forward pass at MAX_LEN (heavy) ---
    if MEASURE_LLM_FLOPS:
        try:
            flops_llm = measure_full_llm_pass_flops(base.encoder, seq_len=MAX_LEN,
                                                    device=device if device.type=="cuda" else "cpu")
            ctx["flops_llm"] = flops_llm
            print(f"[Gemma full pass] {flops_llm['flops_human']} FLOPs | {flops_llm['macs_human']} MACs | Params: {flops_llm['params_human']}")
        except Exception as e:
            ctx["flops_llm"] = {"error": str(e)}
            print(f"[Gemma full pass] FLOPs estimation failed: {e}")

def ev_gemma_tabnsa():
    metrics_val  = evaluate_model(ctx["model"], ctx["val_loader"],  ctx["num_classes"], device)
    metrics_test = evaluate_model(ctx["model"], ctx["test_loader"], ctx["num_classes"], device)
    out = {"val": metrics_val, "test": metrics_test,
           "params": ctx["param_counts"],
           "flops_head": ctx["flops_head"]}
    if "flops_llm" in ctx: out["flops_llm"] = ctx["flops_llm"]
    return out

# ------------------------------------- RUN --------------------------------------
run_with_metrics("GemmaTabNSA", pp_gemma_tabnsa, dl_gemma_tabnsa, tr_gemma_tabnsa, ev_gemma_tabnsa)
# ================================================================================

