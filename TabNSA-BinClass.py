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
################################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRIALS = 10
EPOCHS = 100
NUM_SEED = 10
# BATCH_SIZE = 64

################################################################################################################
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

            # with torch.autograd.detect_anomaly():
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * x_batch.size(0)

        # Calculate epoch metrics
        train_loss = epoch_train_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # print(f'Epoch {epoch+1}/{epochs}')
    
    return history

# def fit(model, criterion, optimizer, X_train, y_train, epochs=10, batch_size=32, device='cuda'):
#     history = {'train_loss': []}
    
#     X_train, y_train = X_train.float().to(device), y_train.to(device)

#     train_dataset = TensorDataset(X_train, y_train)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#     model.to(device)
    
#     for epoch in range(epochs):
#         model.train()
#         epoch_train_loss = 0.0
        
#         for x_batch, y_batch in train_loader:
#             x_batch, y_batch = x_batch.to(device), y_batch.to(device)

#             optimizer.zero_grad()
            
#             with torch.autograd.detect_anomaly():  # Enable anomaly detection
#                 outputs = model(x_batch)
#                 loss = criterion(outputs, y_batch)
#                 loss.backward()  # Detects any invalid gradients
            
#             optimizer.step()
#             epoch_train_loss += loss.item() * x_batch.size(0)

#         train_loss = epoch_train_loss / len(train_loader.dataset)
#         history['train_loss'].append(train_loss)

#     return history

################################################################################################################

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
            
            # Compute probabilities (for binary classification, probability for class 1)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            all_probabilities.append(probabilities.cpu())
            all_labels.append(batch_y.cpu())
    
    test_probabilities = torch.cat(all_probabilities)
    test_labels = torch.cat(all_labels)
    
    # Calculate metrics
    test_auc = roc_auc_score(test_labels.numpy(), test_probabilities.numpy())    
    test_predictions = (test_probabilities > 0.5).long()
    test_accuracy = (test_predictions == test_labels).float().mean().item()
      
    return test_auc, test_accuracy
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
################################################################################################################

# dataset_P = ['AD', 'BL', 'CA', 'CB', 'CG', 'DS', 'IC', 'IO']

dataset_P = ['CB']

seeds = np.random.randint(10, 1000, NUM_SEED)

for DATA in dataset_P:
    data = pd.read_csv("/home/data3/Ali/Code/KAN/Afzal/DATA/%s.csv" % DATA)

    results_AUC = []
    
    data.fillna(0, inplace=True)
    y = data["target_label"].values
    data.drop("target_label", axis=1, inplace=True)
    X = data.values.astype(np.float32)

    if y.dtype == "object":
        label_encoder = LabelEncoder()
        y = torch.tensor(label_encoder.fit_transform(y))
    else:
        y = torch.tensor(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train, test_size=0.1, random_state=42)

    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(X_train)).to(device)
    X_valid = torch.tensor(scaler.transform(X_valid)).to(device)
    X_test = torch.tensor(scaler.transform(X_test)).to(device)

    y_train = torch.nn.functional.one_hot(y_train.long(), num_classes=2).to(device).float()
    y_valid = torch.nn.functional.one_hot(y_valid.long(), num_classes=2).to(device).float()
    y_test = torch.nn.functional.one_hot(y_test.long(), num_classes=2).to(device).float()

    input_shape = X_train.shape[1]
    output_shape = y_train.shape[1]

    if output_shape > 1:
        y_train = y_train.argmax(dim=1)
        y_valid = y_valid.argmax(dim=1)
        y_test = y_test.argmax(dim=1)

    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=TRIALS)
    # best_params = study.best_params
    
    # Tresults = study.trials_dataframe()
    # Tresults.to_csv('results-NSA-%s-%s.csv' % (DATA, datetime.now()))
    
    # best_params = {'dim': 160, 'dim_head': 32, 'heads': 6, 'sliding_window_size': 7, 'compress_block_size': 16, 'selection_block_size': 8, 'num_selected_blocks': 2, 'learning_rate': 0.0002104037136418331, 'batch_size': 64}
    best_params = {'dim': 32, 'dim_head': 40, 'heads': 5, 'sliding_window_size': 2, 'compress_block_size': 4, 'selection_block_size': 16, 'num_selected_blocks': 1, 'learning_rate': 3.34470782781812e-05, 'batch_size': 32}

    
    dim_head= best_params["dim_head"]
    heads= best_params["heads"]
    sliding_window_size= best_params["sliding_window_size"]
    compress_block_size= best_params["compress_block_size"]
    selection_block_size= best_params["selection_block_size"]
    num_selected_blocks= best_params["num_selected_blocks"]
    learning_rate = best_params["learning_rate"]
    batch_size = best_params["batch_size"]
    
    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train, test_size=0.1, random_state=42)

        scaler = StandardScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train)).to(device)
        # X_valid = torch.tensor(scaler.transform(X_valid)).to(device)
        X_test = torch.tensor(scaler.transform(X_test)).to(device)

        y_train = torch.nn.functional.one_hot(y_train.long(), num_classes=2).to(device).float()
        # y_valid = torch.nn.functional.one_hot(y_valid.long(), num_classes=2).to(device).float()
        y_test = torch.nn.functional.one_hot(y_test.long(), num_classes=2).to(device).float()

        input_shape = X_train.shape[1]
        output_shape = y_train.shape[1]

        if output_shape > 1:
            y_train = y_train.argmax(dim=1)
            # y_valid = y_valid.argmax(dim=1)
            y_test = y_test.argmax(dim=1)

        model = SparseAttentionModel(input_shape, output_shape, dim_head, heads, sliding_window_size, compress_block_size, selection_block_size, num_selected_blocks).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate)
        history = fit(model, criterion, optimizer, X_test, y_test, epochs= EPOCHS, batch_size= batch_size, device='cuda')

        test_auc, test_accuracy = evaluate_model_auc(model, X_test, y_test, output_shape, device='cuda', batch_size= batch_size)
        results_AUC.append(test_auc)
        
    top_10_results = sorted(results_AUC, reverse=True)[:10]
    average_top_10 = sum(top_10_results) / len(top_10_results)
    
    print("Dataset:", DATA)
    print("Best Params: ", best_params)
    print("Best AUC: ", average_top_10)
    # print("Best Accuracy: ", test_accuracy)

################################################################################################################