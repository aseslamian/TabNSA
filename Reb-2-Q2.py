## Rebuttal-2 
## Question 2

# NSA-8 with objective function
# It use TabNSA (NSA+TabMixer) parallelly
# It combine the objective function used for TabNSA github code

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch.nn as nn
from scipy.io import arff
from native_sparse_attention_pytorch import SparseAttention
from torch.utils.data import DataLoader, TensorDataset 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from tabmixer import TabMixer 

############################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRIALS = 10
EPOCHS = 100
NUM_SEED = 10
# BATCH_SIZE = 64
seed = 42
############################################################################################################
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
            probabilities = torch.softmax(outputs, dim=1)[:, 1] if output_shape > 1 else outputs.squeeze()
            all_probabilities.append(probabilities.cpu())
            all_labels.append(batch_y.cpu())
    
    test_probabilities = torch.cat(all_probabilities)
    test_labels = torch.cat(all_labels)
    
    # Calculate metrics
    test_auc = roc_auc_score(test_labels.numpy(), test_probabilities.numpy())    
    test_predictions = (test_probabilities > 0.5).long()
    # test_accuracy = (test_predictions == test_labels).float().mean().item()
    
    # print(f'Test AUC: {test_auc:.4f}')
    # print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    
    return test_auc
############################################################################################################
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
         
    return history
############################################################################################################
class TabNSA(nn.Module):
    def __init__(self, input_shape, output_shape, dim_head, heads, sliding_window_size, 
                 compress_block_size, selection_block_size, num_selected_blocks):
        super().__init__()
        
        self.dim = NUM_TOKENS
        self.feature_embedding = nn.Linear(1, self.dim)

        self.attn = SparseAttention(
            dim=self.dim,
            dim_head = dim_head,
            heads = heads,
            sliding_window_size = sliding_window_size,
            compress_block_size = compress_block_size,
            selection_block_size = selection_block_size,
            # compress_block_sliding_stride = compress_block_sliding_stride,
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
        x_1 = self.attn(x)
        x_2 = self.tabmixer(x)
        x = x_1 + x_2
        x = x.mean(dim=1) 
        
        return self.head(x)
############################################################################################################

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

    model = TabNSA(input_shape, output_shape, dim_head, heads, sliding_window_size, compress_block_size, selection_block_size, num_selected_blocks)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # criterion = nn.MSELoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate)
    # history = fit(model, criterion, optimizer, X_train, y_train, epochs= EPOCHS, batch_size= batch_size, device='cuda')
    
    criterion = nn.CrossEntropyLoss() if output_shape > 1 else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = fit(model, criterion, optimizer, X_train, y_train, epochs= EPOCHS, batch_size= BATCH_SIZE, device='cuda')

    
    # y_pred = model(X_valid).cpu().detach().numpy()
    # y_true = y_valid.cpu().detach().numpy()
    # mse = mean_squared_error(y_true, y_pred)
    # rmse = root_mean_squared_error(y_true, y_pred)
    # r2  = r2_score(y_true, y_pred)
    # combined_objective = mse + 100 * abs(r2 - 1)
    
    test_auc = evaluate_model_auc(model, X_test, y_test, output_shape, device='cuda', batch_size= batch_size)
    
    return test_auc
################################################################################################################

# DATA = 'cpu'  # usr
# # DATA = 'topo'   # oz267
# # DATA = 'Moneyball'  # RSquared
# # DATA = 'sarcos'  # V28

# data = arff.loadarff("/home/data3/Ali/Code/KAN/Regression/Dataset/%s.arff" % DATA)
# data = pd.DataFrame(data[0])

# data.fillna(0, inplace=True)
# y = data["usr"].values
# data.drop("usr", axis=1, inplace=True)
# X = data.values.astype(np.float32)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
############################################################################################################

dataset_P = ['AD', 'BL', 'CA', 'CB', 'CG', 'DS', 'IC', 'IO']
DATA = dataset_P[3]
print(DATA)

data = pd.read_csv("/home/data3/Ali/Code/KAN/Afzal/DATA/%s.csv" % DATA) 
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
############################################################################################################


scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train)).to(device)
X_valid = torch.tensor(scaler.transform(X_valid)).to(device)
X_test = torch.tensor(scaler.transform(X_test)).to(device)

# y_train = torch.tensor(y_train).to(device).float().view(-1, 1)
# y_valid = torch.tensor(y_valid).to(device).float().view(-1, 1)
# y_test = torch.tensor(y_test).to(device).float().view(-1, 1)

y_train = y_train.detach().clone().to(device).float().view(-1, 1)
y_valid = y_valid.detach().clone().to(device).float().view(-1, 1)
y_test  = y_test.detach().clone().to(device).float().view(-1, 1)


input_shape = X_train.shape[1]
output_shape = 1  # Regression

EPOCHS = 50
NUM_TOKENS = 64
BATCH_SIZE = 32
NUM_FEATURES = input_shape
NUM_CLASSES = output_shape

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=TRIALS)
# best_params = study.best_params

best_params = {'dim': 64, 'dim_head': 64, 'heads': 8, 'sliding_window_size': 2, 'compress_block_size': 4, 
               'selection_block_size': 4, 'compress_block_sliding_stride': 2, 'num_selected_blocks': 2, 'learning_rate': 8.34869727747217e-05, 'batch_size': 96}

dim_head= best_params["dim_head"]
heads= best_params["heads"]
sliding_window_size= best_params["sliding_window_size"]
compress_block_size= best_params["compress_block_size"]
selection_block_size= best_params["selection_block_size"]
num_selected_blocks= best_params["num_selected_blocks"]
learning_rate = best_params["learning_rate"]
compress_block_sliding_stride = 2
batch_size = best_params["batch_size"]


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# model = SparseAttentionModel(input_shape, output_shape, dim_head, heads, sliding_window_size, compress_block_size, selection_block_size, num_selected_blocks).to(device)
# criterion = nn.CrossEntropyLoss()

model = TabNSA(input_shape, output_shape, dim_head, heads, sliding_window_size, compress_block_size, selection_block_size, num_selected_blocks)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

criterion = nn.CrossEntropyLoss() if output_shape > 1 else nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate)

def to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x

X_np = to_numpy(X_train)
y_np = to_numpy(y_train)
y_np = to_numpy(y_train).ravel()


corr = np.abs(np.corrcoef(X_np, y_np, rowvar=False)[-1, :-1])
order = np.argsort(-corr)

X_train_corr = X_train[:, order]
X_test_corr  = X_test[:, order]

def run_experiment(X_train_used, X_test_used, label):
    results = []

    for i in range(TRIALS):
        print(f"[{label}] Trial {i+1}/{TRIALS}")

        # IMPORTANT: reinitialize model & optimizer every trial
        model = TabNSA(
            input_shape,
            output_shape,
            dim_head,
            heads,
            sliding_window_size,
            compress_block_size,
            selection_block_size,
            num_selected_blocks
        ).to(device)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss() if output_shape > 1 else nn.MSELoss()

        fit(
            model,
            criterion,
            optimizer,
            X_train_used,
            y_train,
            epochs=EPOCHS,
            batch_size=batch_size,
            device=device
        )

        # with torch.no_grad():
        #     y_pred = model(X_test_used).cpu().numpy()
        #     y_true = y_test.cpu().numpy()

        # mse  = mean_squared_error(y_true, y_pred)
        # mae  = mean_absolute_error(y_true, y_pred)
        # rmse = root_mean_squared_error(y_true, y_pred)
        # r2   = r2_score(y_true, y_pred)
        # results.append([mse, mae, rmse, r2])
        # print(f"  MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        
        test_auc = evaluate_model_auc(model, X_test_used, y_test, output_shape, device='cuda', batch_size= batch_size)
        results.append([test_auc])
        print(f"  AUC={test_auc:.4f}")

    results = np.array(results)

    # print(f"\n[{label}] Mean ± Std over {TRIALS} trials")
    # print(f"MSE : {results[:,0].mean():.4f} ± {results[:,0].std():.4f}")
    # print(f"MAE : {results[:,1].mean():.4f} ± {results[:,1].std():.4f}")
    # print(f"RMSE: {results[:,2].mean():.4f} ± {results[:,2].std():.4f}")
    # print(f"R2  : {results[:,3].mean():.4f} ± {results[:,3].std():.4f}")

    print(f"\n[{label}] Mean ± Std over {TRIALS} trials")
    print(f"AUC : {results[:,0].mean():.4f} ± {results[:,0].std():.4f}")

    return results

# ---------------- index-based adjacency ----------------
X_train_idx = X_train
X_test_idx  = X_test

res_index = run_experiment(X_train_idx, X_test_idx, "Index-based adjacency")

# ---------------- correlation-based adjacency ----------------
X_train_corr = X_train[:, order]
X_test_corr  = X_test[:, order]

res_corr = run_experiment(X_train_corr, X_test_corr, "Correlation-based adjacency")
# ---------------------------------------------------------

import numpy as np

# def summarize_results(results, name):
#     results = np.array(results)

#     mean = results.mean(axis=0)
#     var  = results.var(axis=0)
#     std  = results.std(axis=0)

#     metrics = ["MSE", "MAE", "RMSE", "R2"]

#     print(f"\n{name}")
#     for i, m in enumerate(metrics):
#         print(f"{m}: mean = {mean[i]:.4f}, var = {var[i]:.4f}, std = {std[i]:.4f}")

#     return mean, var, std

def summarize_results(results, label):
    results = np.array(results)

    mean = results.mean(axis=0)
    var  = results.var(axis=0)
    std  = results.std(axis=0)

    print(f"\n[{label}] Mean ± Std over {TRIALS} trials")
    print(f"AUC : {mean[0]:.4f} ± {std[0]:.4f}")

    return mean, var, std

mean_idx, var_idx, std_idx = summarize_results(res_index, "Index-based adjacency")
mean_corr, var_corr, std_corr = summarize_results(res_corr, "Correlation-based adjacency")
