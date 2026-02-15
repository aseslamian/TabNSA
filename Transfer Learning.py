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
import os
import tabmixer
from tabmixer import TabMixer
################################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRIALS = 2
EPOCHS = 5
NUM_SEED = 50
BATCH_SIZE = 32
NUM_CLASSES = 2

################################################################################################################
def fit(model, criterion, optimizer, X_train, y_train, epochs=10, batch_size= BATCH_SIZE, device='cuda'):
    history = {'train_loss': []}
    
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
        train_loss = epoch_train_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)

    return history
################################################################################################################
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
            dim_tokens=NUM_FEATURES,       
            dim_features=self.dim,    
            dim_feedforward=256 
        )

        self.head = nn.Sequential(
            nn.Linear(self.dim, 32),
            nn.GELU(),
            nn.Linear(32, NUM_CLASSES)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        x = x.unsqueeze(-1)
        x = self.feature_embedding(x)
        x_1 = self.attention(x)
        x_2 = self.tabmixer(x)
        x = x_1 + x_2
        x = x.mean(dim=1) 
        
        return self.head(x) 

################################################################################################################
def load_and_preprocess_data(path, path2, device, num_classes=2, seed=42):

    # --- Load the source dataset ---
    data = pd.read_csv(path)
    data.fillna(0, inplace=True)
    
    # Separate target and features from source data
    y1 = data["target_label"].values
    data.drop("target_label", axis=1, inplace=True)
    X1 = data.values.astype(np.float32)
    source_columns = data.columns.tolist()
    
    data2 = pd.read_csv(path2)
    data2.fillna(0, inplace=True)
    
    y2 = data2["target_label"].values
    data2.drop("target_label", axis=1, inplace=True)
    X2 = data2.values.astype(np.float32)
    
    missing_cols_target = set(source_columns) - set(data2.columns)
    for col in missing_cols_target:
        data2[col] = 0
    data2 = data2[source_columns]
    X2 = data2.values.astype(np.float32)
    
    if y1.dtype == "object":
        label_encoder = LabelEncoder()
        y1 = label_encoder.fit_transform(y1)
    else:
        label_encoder = None
    y1 = torch.tensor(y1)
    
    # For the target dataset: if label_encoder exists, use it; otherwise, fit a new one.
    if y2.dtype == "object":
        if label_encoder is not None:
            y2 = label_encoder.transform(y2)
        else:
            label_encoder = LabelEncoder()
            y2 = label_encoder.fit_transform(y2)
    y2 = torch.tensor(y2)

    # Split data: first into train and test, then split a validation set from train
    X_train, X_valid, y_train, y_valid = train_test_split(X1, y1, stratify=y1, test_size=0.2, random_state=seed)

    # Initialize and apply the scaler on features
    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(X_train)).to(device)
    X_valid = torch.tensor(scaler.transform(X_valid)).to(device)

    # One-hot encode the labels
    y_train = torch.nn.functional.one_hot(y_train.long(), num_classes=num_classes).to(device).float()
    y_valid = torch.nn.functional.one_hot(y_valid.long(), num_classes=num_classes).to(device).float()

    dataset = {
            "X_train": X_train,
            "X_valid": X_valid,
            "y_train": y_train,
            "y_valid": y_valid,       
        }
    
    ############################################################
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, stratify=y2, test_size=0.2, random_state=42)

    # Initialize and apply the scaler on features
    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(X_train)).to(device)
    X_test  = torch.tensor(scaler.transform(X_test)).to(device)

    # One-hot encode the labels
    y_train = torch.nn.functional.one_hot(y_train.long(), num_classes=num_classes).to(device).float()
    y_test  = torch.nn.functional.one_hot(y_test.long(), num_classes=num_classes).to(device).float()

    # Determine shapes
    input_shape = X_train.shape[1]
    output_shape = y_train.shape[1]

    dataset2 = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,        
        }

    return dataset, dataset2, source_columns, data2.columns, label_encoder

###############################################################################################################
def evaluate_model_auc(model, X_test, y_test, output_shape, device='cuda', batch_size=BATCH_SIZE):
    from sklearn.metrics import roc_auc_score
    import torch.nn.functional as F
    
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

            # Apply softmax across the class dimension
            probabilities = F.softmax(outputs, dim=1)
            all_probabilities.append(probabilities.cpu())
            all_labels.append(batch_y.cpu())

    test_probabilities = torch.cat(all_probabilities).numpy()
    test_labels = torch.cat(all_labels).numpy()

    return roc_auc_score(test_labels, test_probabilities, multi_class='ovr')
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

    model = TabNSA(input_shape, output_shape, dim_head, heads, sliding_window_size, compress_block_size, selection_block_size, num_selected_blocks).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate)
    history = fit(model, criterion, optimizer, X_train, y_train, epochs= EPOCHS, batch_size= batch_size, device='cuda')
    test_auc = evaluate_model_auc(model, X_valid, y_valid, output_shape, device='cuda', batch_size= batch_size)

    return test_auc
################################################################################################################

# dataset_P = ['Credit-g(CG)', 'Adult(AD)' , 'Blastchar(BL)', 'Credit-approval(CA)', 'cylinder-bands(CB)', 'dress-sale(DS)','income (IC)', 'insurance+company(IO)']

dataset_P = ['Credit-g(CG)', 'Blastchar(BL)']


for DATA in dataset_P:
    path = f"/home/data3/Ali/Code/KAN/TL-Test/DATA/{DATA}/Preprocessed"
    # path = f"/mnt/gpfs2_4m/scratch/aes255/Afzal/run_Cheb/DATA/{DATA}/Preprocessed"


    parent_path = os.path.dirname(path)
    target_path = os.path.join(parent_path, 'TransferLearning')

    DATA1_PATH = os.path.join(target_path, 'data1', 'data_processed.csv')
    DATA2_PATH = os.path.join(target_path, 'data2', 'data_processed.csv')

    dataset, dataset2, source_columns, target_columns, label_encoder = load_and_preprocess_data(DATA1_PATH, DATA2_PATH, device, num_classes=2)

    input_shape = dataset['X_train'].shape[1]
    output_shape = dataset['y_train'].shape[1]

    NUM_FEATURES = input_shape
    NUM_TOKENS = 64
    NUM_CLASSES = output_shape

    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_valid = dataset['X_valid']
    y_valid = dataset['y_valid']

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=TRIALS)
    best_params = study.best_params

    X_fine = dataset2['X_train']
    y_fine = dataset2['y_train']
    X_test = dataset2['X_test']
    y_test = dataset2['y_test']

    dim_head= best_params["dim_head"]
    heads= best_params["heads"]
    sliding_window_size= best_params["sliding_window_size"]
    compress_block_size= best_params["compress_block_size"]
    selection_block_size= best_params["selection_block_size"]
    num_selected_blocks= best_params["num_selected_blocks"]
    learning_rate = best_params["learning_rate"]
    batch_size = best_params["batch_size"]

    model = TabNSA(input_shape, output_shape, dim_head, heads, sliding_window_size, compress_block_size, selection_block_size, num_selected_blocks).to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate)

    history = fit(model, criterion, optimizer, X_fine, y_fine, epochs= EPOCHS, batch_size= batch_size, device='cuda')

    test_auc = evaluate_model_auc(model, X_test, y_test, output_shape, device='cuda', batch_size= batch_size)
    result = {
        "dataset": DATA,
        "best_params": best_params,
        "test_auc": test_auc
    }
    # Save the results
    results = []
    results.append(result)
    # print("Dataset: ", DATA)   
    # print("Best Params: ", best_params)
    # print("Test AUC: ", test_auc)

print("Results: ", results)
results_df = pd.DataFrame(results)
results_df.to_csv(f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)