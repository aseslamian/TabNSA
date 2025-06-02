## Code for GemmaTabNSA
### Sample Few-Shot Learning
# This code are Zero-Shot on Samples

import math
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
import glob
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
from tqdm import tqdm
from torch.optim import LBFGS
import torch.nn as nn
from scipy.io import arff
from native_sparse_attention_pytorch import SparseAttention
from torch.utils.data import DataLoader, TensorDataset 
import os
import tabmixer
from tabmixer import TabMixer
from transformers import BertTokenizer
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import BertModel
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM

# import sys
# !{sys.executable} -m pip install transformers

from huggingface_hub import login
login("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")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_FEATURES = 64 
TRIAL = 5
EPOTCHES = 80

##########################################################################################################
def load_dataset_auto(file):
    df = file
    feature_cols = df.columns[:-1].tolist()  
    label_col = df.columns[-1]            
    return df, feature_cols, label_col
def row_to_prompt(row, feature_cols):
    return ", ".join([f"{col} is {row[col]}" for col in feature_cols])
def prepare_prompt_data(filepath):
    df, feature_cols, label_col = load_dataset_auto(filepath)
    df["prompt"] = df.apply(lambda row: row_to_prompt(row, feature_cols), axis=1)
    df["label"] = df[label_col]
    return df[["prompt", "label"]]

##########################################################################################################
tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it")
llm_model = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-2b-it", device_map="auto")
HIDDEN_SIZE = llm_model.config.hidden_size

def tokenize_dataset(df):
    return tokenizer(
        df["prompt"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
##########################################################################################################
class PromptDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }
        
##########################################################################################################       
class TabNSA(nn.Module):
    def __init__(self, input_shape, output_shape, dim_head, heads, sliding_window_size, compress_block_size, selection_block_size, num_selected_blocks):
        super().__init__()
        
        self.dim = 64
        self.feature_embedding = nn.Linear(1, self.dim)

        self.attention = SparseAttention(
            dim=self.dim,
            dim_head = dim_head,
            heads = heads,
            sliding_window_size = sliding_window_size,
            compress_block_size = compress_block_size,
            selection_block_size = selection_block_size,
            num_selected_blocks = num_selected_blocks
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

    def forward(self, x):
        x = x.unsqueeze(-1) 
        x = self.feature_embedding(x) 
        x_1 = self.attention(x)
        x_2 = self.tabmixer(x)
        x = x_1 + x_2
        x = x.mean(dim=1) 
        return self.head(x)
    
##########################################################################################################  
class GemmaTabNSA(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size, llm_model, **tabnsa_params):
        super().__init__()
        self.encoder = llm_model
        self.adapter = nn.Linear(hidden_size, num_features)
        self.tabnsa = TabNSA(
            input_shape=num_features,
            output_shape=num_classes,
            **tabnsa_params
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            pooled = (hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)

        adapted = self.adapter(pooled)
        return self.tabnsa(adapted)
    
##########################################################################################################   
def fit(model, criterion, optimizer, train_loader, device, epochs=3, val_loader=None, patience=5):
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    model.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * input_ids.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)
        history["train_loss"].append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}", end="")

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * input_ids.size(0)
            avg_val_loss = val_loss / len(val_loader.dataset)
            history["val_loss"].append(avg_val_loss)
            print(f" | Val Loss: {avg_val_loss:.4f}")
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
        else:
            print()
    return history

##########################################################################################################
class GemmaTabNSA(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size, llm_model, **tabnsa_params):
        super().__init__()
        self.encoder = llm_model
        self.adapter = nn.Linear(hidden_size, num_features)
        self.tabnsa = TabNSA(
            input_shape=num_features,
            output_shape=num_classes,
            **tabnsa_params
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            pooled = (hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)

        adapted = self.adapter(pooled)
        return self.tabnsa(adapted)

# class GemmaTabNSA(nn.Module):
#     def __init__(self, num_features, num_classes, hidden_size, llm_model, **tabnsa_params):
#         super().__init__()
#         self.encoder = llm_model
#         self.adapter = nn.Linear(hidden_size, num_features)
#         self.tabnsa = TabNSA(
#             input_shape=num_features,
#             output_shape=num_classes,
#             **tabnsa_params
#         )

#     def forward(self, input_ids, attention_mask):
#         with torch.no_grad():
#             outputs = self.encoder(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 output_hidden_states=True
#             )
#             hidden = outputs.hidden_states[-1]
#             pooled = hidden[:, 0, :]    
#         adapted = self.adapter(pooled)  
#         return self.tabnsa(adapted)
    
##########################################################################################################
def evaluate_model(model, val_loader, num_classes, device, return_auc=False):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)

    if return_auc and num_classes == 2:
        auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
        return acc, auc
    elif return_auc and num_classes > 2:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovo')
        return acc, auc
    else:
        return acc
    
# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
##########################################################################################################

# DATA = "Credit-g(CG)"
# DATA = "Credit-approval(CA)"
# DATA = "cylinder-bands(CB)"
# DATA = "dress-sale(DS)"
# DATA = "Adult(AD)"
# DATA = "insurance+company(IO)"
# DATA = "Blastchar(BL)"

# DATA = "Blood"
# DATA = "California"
# DATA = "Diabetes"
# DATA = "Heart"
DATA = "Jungle"

seeds = np.random.randint(0, 500, size=20)
# dataset_P = ['Bank','Diabetes']
splits = [16, 32, 64, 128, 256, 512]


seed = 42 
split = 128

results = []


# torch.manual_seed(seed)
# np.random.seed(seed)

DATA1_PATH = f"/jet/home/eslamian/Data/{DATA}/*.csv"
csv_files = glob.glob(DATA1_PATH)
df_list = [pd.read_csv(file) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)

# train_frac = 0.09
# split_index = int(len(df) * train_frac)

split_index = split
train_df = df[:split_index]
test_df = df[split_index:]

F1P1 = prepare_prompt_data(train_df.copy())
F1P2 = prepare_prompt_data(test_df.copy())
le = LabelEncoder()
le.fit(F1P1["label"]) 
for df in [F1P1, F1P2]:
    df["label_id"] = le.transform(df["label"])

NUM_CLASSES = len(le.classes_) 
print(NUM_CLASSES)
le = LabelEncoder()
F1P1["label_id"] = le.fit_transform(F1P1["label"])
F1P2["label_id"] = le.transform(F1P2["label"])

# Tokenize
train_enc = tokenize_dataset(F1P1)
val_enc   = tokenize_dataset(F1P2)

# Datasets
few_dataset = PromptDataset(train_enc, F1P1["label_id"].tolist())
test_dataset   = PromptDataset(val_enc,   F1P2["label_id"].tolist())

best_params = {
    "dim_head": 56,
    "heads": 4,
    "sliding_window_size": 4,
    "compress_block_size": 12,
    "selection_block_size": 8,
    "num_selected_blocks": 3,
    "learning_rate": 0.0004265095743358644,
    "batch_size": 128
}
print("Best parameters:", best_params)
learning_rate = best_params["learning_rate"]
batch_size = best_params["batch_size"]
best_params = {k: v for k, v in best_params.items() if k not in ["learning_rate", "batch_size"]}
model = GemmaTabNSA(num_features=NUM_FEATURES, num_classes=NUM_CLASSES, hidden_size=HIDDEN_SIZE, llm_model=llm_model, **best_params).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

few_loader = DataLoader(few_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

fit(model, criterion, optimizer, few_loader, device, epochs=EPOTCHES)
acc, auc = evaluate_model(model, test_loader, num_classes=NUM_CLASSES, device=device, return_auc=True)
# results.append(auc)
print(f"Running for DATA: {DATA}, split: {split}, AUC: {auc:.4f}")

        # results_sorted = sorted(results, reverse=True)
        # top5 = results_sorted[:5]

        # mean_auc = np.mean(top5)
        # std_auc = np.std(top5)
        # results.append(auc)
        # print(f"Running for DATA: {DATA}, split: {split}, Mean AUC over {len(top5)} seeds: {mean_auc:.4f} ± {std_auc:.4f}")
        # # print(f"Running for DATA: {DATA}, split: {split}, AUC: {auc:.4f}")


# results_sorted = sorted(results, reverse=True)
# top5 = results_sorted[:5]

# mean_auc = np.mean(top5)
# std_auc = np.std(top5)
# print(f"Mean AUC over {len(top5)} seeds: {mean_auc:.4f} ± {std_auc:.4f}")
