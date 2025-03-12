import math
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
from scipy.io import arff
from sklearn.metrics import roc_auc_score, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRIALS = 10
EPOCHS = 100
NUM_SEED = 10
# BATCH_SIZE = 64

def fit(model, criterion, optimizer, X_train, y_train, epochs=10, batch_size=32, device='cuda'):
    
    history = {'train_loss': [], 'val_loss': []}
    # print ("Batch size: ", batch_size)
    # Convert to float32 once and move to device
    X_train, y_train = X_train.float().to(device), y_train.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.to(device)
    
    for epoch in range(epochs):
        
        model.train()
        epoch_train_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * x_batch.size(0)

        # Calculate epoch metrics
        train_loss = epoch_train_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # print(f'Epoch {epoch+1}/{epochs}')
    
    return history

class SparseAttentionModel(nn.Module):
    def __init__(self, input_shape, output_shape, dim_head, heads, sliding_window_size, 
                 compress_block_size, selection_block_size, num_selected_blocks):
        super().__init__()
        
        self.dim = 64  # Embedding dimension for features
        
        # Project scalar features to embedding space
        self.feature_embedding = nn.Linear(1, self.dim)
        
        # Sparse attention module
        self.attention = SparseAttention(
            dim=self.dim,
            dim_head = dim_head,
            heads = heads,
            sliding_window_size = sliding_window_size,
            compress_block_size = compress_block_size,
            selection_block_size = selection_block_size,
            num_selected_blocks = num_selected_blocks
        )
        
        # Classification/regression head
        self.head = nn.Sequential(
            nn.Linear(self.dim, 32),
            nn.GELU(),
            nn.Linear(32, output_shape)
        )

    def forward(self, x):

        batch_size = x.shape[0]
        x = x.unsqueeze(-1) 
        x = self.feature_embedding(x)
        x = self.attention(x)  # (batch, num_features, dim)
        x = x.mean(dim=1)  # (batch, dim)
        return self.head(x)
    
################################################################################################################
def evaluate_model_auc(model, X_test, y_test, output_shape, device='cuda', batch_size=64):
    
    # Create test dataset and loader
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model.to(device)
    model.eval()
    
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            probabilities = torch.softmax(outputs, dim=1)
            all_probabilities.append(probabilities.cpu())
            all_labels.append(batch_y.cpu())
    
    # Concatenate results
    test_probabilities = torch.cat(all_probabilities).numpy()  # Shape: (n_samples, n_classes)
    test_labels = torch.cat(all_labels).numpy()  # Shape: (n_samples,)

    # Compute AUC-OVO (One-Versus-One)
    try:
        test_auc_ovo = roc_auc_score(test_labels, test_probabilities, multi_class='ovo')
    except ValueError:
        test_auc_ovo = None  # Handle case where AUC can't be computed

    # Compute Predictions
    test_predictions = test_probabilities.argmax(axis=1)  # Get class with highest probability

    # Compute Accuracy
    test_accuracy = accuracy_score(test_labels, test_predictions)

    # Compute F1-score (weighted for multi-class classification)
    test_f1 = f1_score(test_labels, test_predictions, average='weighted')

    return test_auc_ovo, test_accuracy, test_f1
################################################################################################################

def objective(trial):

    dim = trial.suggest_int("dim", 32, 256, step=32)  
    dim_head = trial.suggest_int("dim_head", 8, 64, step=8)
    heads = trial.suggest_int("heads", 1, 8)
    sliding_window_size = trial.suggest_int("sliding_window_size", 1, 8)
    compress_block_size = trial.suggest_int("compress_block_size", 4, 16, step=4)
    selection_block_size = trial.suggest_int("selection_block_size", 4, 16, step=4)
    num_selected_blocks = trial.suggest_int("num_selected_blocks", 1, 4)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    # dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)

    model = SparseAttentionModel(input_shape, output_shape, dim_head, heads, sliding_window_size, compress_block_size, selection_block_size, num_selected_blocks).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate)
    history = fit(model, criterion, optimizer, X_train, y_train, epochs= EPOCHS, batch_size= batch_size, device='cuda')
    
    test_auc, test_accuracy = evaluate_model_auc(model, X_valid, y_valid, output_shape, device='cuda', batch_size= batch_size)

    return test_auc

DATA = 'ForestCovertype'

data = arff.loadarff("/your path directory/Multi-Class/%s.arff" % DATA)
data = pd.DataFrame(data[0])

target_column = data.columns[-1]
data[target_column] = data[target_column].str.decode("utf-8")

label_encoder = LabelEncoder()
data[target_column] = label_encoder.fit_transform(data[target_column])

# Load dataset
le = LabelEncoder()

# Decode categorical columns if needed (ARFF stores them as bytes)
for col in data.select_dtypes([np.object_, "object"]).columns:
    data[col] = data[col].str.decode("utf-8")
    data[col] = le.fit_transform(data[col])

# Separate target and features
target = 'class'  # Ensure this is correct
X = data.drop(columns=[target])
y = data[target]

data.fillna(0, inplace=True)
y = data["class"].values
data.drop("class", axis=1, inplace=True)
X = data.values.astype(np.float32)
################################################################################################################

if y.dtype == "object":
    label_encoder = LabelEncoder()
    y = torch.tensor(label_encoder.fit_transform(y))
else:
    y = torch.tensor(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train, test_size=0.1, random_state=42)

scaler = StandardScaler()
# X_temp = torch.tensor(scaler.fit_transform(X_temp)).to(device)
X_train = torch.tensor(scaler.fit_transform(X_train)).to(device)
X_valid = torch.tensor(scaler.transform(X_valid)).to(device)
X_test = torch.tensor(scaler.transform(X_test)).to(device)

# y_temp = torch.nn.functional.one_hot(y_temp.long(), num_classes=7).to(device).float()
y_train = torch.nn.functional.one_hot(y_train.long(), num_classes=7).to(device).float()
y_valid = torch.nn.functional.one_hot(y_valid.long(), num_classes=7).to(device).float()
y_test = torch.nn.functional.one_hot(y_test.long(), num_classes=7).to(device).float()

input_shape = X_train.shape[1]
output_shape = y_train.shape[1]

# if output_shape > 1:
#     y_train = y_train.argmax(dim=1)
#     y_valid = y_valid.argmax(dim=1)
#     y_test = y_test.argmax(dim=1)

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=TRIALS)
# best_params = study.best_params

# Tresults = study.trials_dataframe()
# Tresults.to_csv('results-NSA-%s-%s.csv' % (DATA, datetime.now()))

best_params = {'dim': 160, 'dim_head': 32, 'heads': 6, 'sliding_window_size': 7, 'compress_block_size': 16, 'selection_block_size': 8, 'num_selected_blocks': 2, 'learning_rate': 0.0002104037136418331, 'batch_size': 64}

dim_head= best_params["dim_head"]
heads= best_params["heads"]
sliding_window_size= best_params["sliding_window_size"]
compress_block_size= best_params["compress_block_size"]
selection_block_size= best_params["selection_block_size"]
num_selected_blocks= best_params["num_selected_blocks"]
learning_rate = best_params["learning_rate"]
batch_size = best_params["batch_size"]

# scaler = StandardScaler()
# X_train = torch.tensor(scaler.fit_transform(X_train)).to(device)
# # X_valid = torch.tensor(scaler.transform(X_valid)).to(device)
# X_test = torch.tensor(scaler.transform(X_test)).to(device)

# y_train = torch.nn.functional.one_hot(y_train.long(), num_classes=7).to(device).float()
# # y_valid = torch.nn.functional.one_hot(y_valid.long(), num_classes=2).to(device).float()
# y_test = torch.nn.functional.one_hot(y_test.long(), num_classes=7).to(device).float()

input_shape = X_train.shape[1]
output_shape = y_train.shape[1]

if output_shape > 1:
    y_train = y_train.argmax(dim=1)
    # y_valid = y_valid.argmax(dim=1)
    y_test = y_test.argmax(dim=1)

model = SparseAttentionModel(input_shape, output_shape, dim_head, heads, sliding_window_size, compress_block_size, selection_block_size, num_selected_blocks).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate)
history = fit(model, criterion, optimizer, X_train, y_train, epochs= EPOCHS, batch_size= batch_size, device='cuda')

auc_ovo, accuracy, f1 = evaluate_model_auc(model, X_test, y_test, output_shape)
print(f"Test AUC-OVO: {auc_ovo:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test F1-Score: {f1:.4f}")

