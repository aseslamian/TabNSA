# Run on Broka and KAn environments

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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

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
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * x_batch.size(0)

        # Calculate epoch metrics
        train_loss = epoch_train_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # print(f'Epoch {epoch+1}/{epochs}')
    
    return history
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

def objective(trial):
   
    dim = trial.suggest_int("dim", 32, 256, step=32)
    dim_head = trial.suggest_int("dim_head", 8, 64, step=8)
    heads = trial.suggest_int("heads", 1, 8)
    sliding_window_size = trial.suggest_int("sliding_window_size", 1, 8)
    compress_block_size = trial.suggest_int("compress_block_size", 4, 16, step=4)
    selection_block_size = trial.suggest_int("selection_block_size", 2, compress_block_size, step=2)
    # selection_block_size = trial.suggest_int("selection_block_size", 4, 8, step=2)
    num_selected_blocks = trial.suggest_int("num_selected_blocks", 1, 4)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    # dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)

    model = SparseAttentionModel(input_shape, output_shape, dim_head, heads, sliding_window_size, compress_block_size, selection_block_size, num_selected_blocks).to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate)
    history = fit(model, criterion, optimizer, X_train, y_train, epochs= EPOCHS, batch_size= batch_size, device='cuda')
    
    y_pred = model(X_valid).cpu().detach().numpy()
    y_true = y_valid.cpu().detach().numpy()
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    combined_objective = mse + 100 * abs(r2 - 1)
    
    return combined_objective
################################################################################################################

seed = 42

# DATA = 'cpu'  # usr
# DATA = 'topo'   # oz267
# DATA = 'Moneyball'  # RSquared
DATA = 'sarcos'  # V28

data = arff.loadarff("/your path directory/Dataset/%s.arff" % DATA)
data = pd.DataFrame(data[0])

data.fillna(0, inplace=True)
y = data["V28"].values
data.drop("V28", axis=1, inplace=True)
X = data.values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train)).to(device)
X_valid = torch.tensor(scaler.transform(X_valid)).to(device)
X_test = torch.tensor(scaler.transform(X_test)).to(device)

y_train = torch.tensor(y_train).to(device).float().view(-1, 1)
y_valid = torch.tensor(y_valid).to(device).float().view(-1, 1)
y_test = torch.tensor(y_test).to(device).float().view(-1, 1)

input_shape = X_train.shape[1]
output_shape = 1  # Regression

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=TRIALS)
# best_params = study.best_params

best_params = {'dim': 64, 'dim_head': 48, 'heads': 1, 'sliding_window_size': 6, 'compress_block_size': 12, 'selection_block_size': 2, 'num_selected_blocks': 2, 'learning_rate': 8.34869727747217e-05, 'batch_size': 96}


dim_head= best_params["dim_head"]
heads= best_params["heads"]
sliding_window_size= best_params["sliding_window_size"]
compress_block_size= best_params["compress_block_size"]
selection_block_size= best_params["selection_block_size"]
num_selected_blocks= best_params["num_selected_blocks"]
learning_rate = best_params["learning_rate"]
batch_size = best_params["batch_size"]


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


model = SparseAttentionModel(input_shape, output_shape, dim_head, heads, sliding_window_size, compress_block_size, selection_block_size, num_selected_blocks).to(device)
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate)
history = fit(model, criterion, optimizer, X_train, y_train, epochs= EPOCHS, batch_size= batch_size, device='cuda')

y_pred = model(X_test).cpu().detach().numpy()
y_true = y_test.cpu().detach().numpy()

# Calculate regression metrics
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)
r2  = r2_score(y_true, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R^2 Score: {r2:.4f}")
