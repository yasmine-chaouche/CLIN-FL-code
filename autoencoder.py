import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import copy
import wandb
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import utils

class Autoencoder_CICIDS2017(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder_CICIDS2017, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Autoencoder_5G3E(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder_5G3E, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU()
        )
                
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    


def train_local_model(seed,optimizer, model, train_loader, val_loader, device, local_epochs):
    utils.set_random_seeds(seed)
    criterion = nn.MSELoss()
    model.to(device)
    
    last_train_loss = 0.0
    last_val_loss = 0.0

    for epoch in range(local_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            data = batch[0].to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        last_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                data = batch[0].to(device)
                output = model(data)
                loss = criterion(output, data)
                val_loss += loss.item()
        last_val_loss = val_loss / len(val_loader)

    return model.state_dict(), last_train_loss, last_val_loss

def federated_averaging(state_dicts, sample_counts):
    total_samples = sum(sample_counts)
    avg_state_dict = copy.deepcopy(state_dicts[0])
    
    for key in avg_state_dict.keys():
        # Weighted sum for each parameter
        weighted_params = sum(state_dicts[i][key] * (sample_counts[i] / total_samples) 
                              for i in range(len(state_dicts)))
        avg_state_dict[key] = weighted_params
    return avg_state_dict

def get_reconstruction_error(model, loader, y_test, device):
    model.eval()
    model.to(device)
    errors = []
    with torch.no_grad():
        for batch in loader:
            data = batch[0].to(device)
            reconstruction = model(data)
            # MSE per sample: mean((orig - recon)^2, dim=1)
            batch_error = torch.mean((data - reconstruction)**2, dim=1)
            errors.extend(batch_error.cpu().numpy())
    
    return pd.DataFrame({
        'error': errors,
        'y_true': y_test
    }).reset_index(drop=True)

def federated_learning(seed,dataset,localEpochs,batch_size,learningRate, nb_clients, nb_rounds, split_type, x_val_loader, x_test_loader, y_test):
    utils.set_random_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    client_files = [f'./{split_type}_split/AIF_{i}_{split_type}.csv' for i in range(nb_clients)]
    
    # Initialize Global Model
    sample_df = pd.read_csv(client_files[0])
    input_dim = sample_df.shape[1]
    if dataset == "CICIDS2017":
        global_model = Autoencoder_CICIDS2017(input_dim)
    elif dataset == "5G3E":
        global_model = Autoencoder_5G3E(input_dim)
    
    for rnd in range(nb_rounds):
        print(f"\nRound {rnd+1}/{nb_rounds}")
        local_state_dicts = []
        sample_counts = []
        local_train_losses = []
        local_val_losses = []
        
        for client_file in client_files:
            # Load local client data
            df_client = pd.read_csv(client_file)
            num_samples = len(df_client)
            sample_counts.append(num_samples)
            
            # Prepare PyTorch DataLoader for client
            client_tensor = torch.tensor(df_client.values, dtype=torch.float32)
            client_loader = DataLoader(TensorDataset(client_tensor), batch_size=batch_size, shuffle=True)
            
            # Local Training
            if dataset == "CICIDS2017":
                local_model = Autoencoder_CICIDS2017(input_dim)
                local_model.load_state_dict(global_model.state_dict())
                optimizer = optim.NAdam(local_model.parameters(),lr=learningRate)
            elif dataset == "5G3E":
                local_model = Autoencoder_5G3E(input_dim)
                local_model.load_state_dict(global_model.state_dict())
                optimizer = optim.Adam(local_model.parameters(),lr=learningRate)
            state_dict, train_loss, val_loss = train_local_model(seed,
                    optimizer, local_model, client_loader, x_val_loader, device, localEpochs
            )
            
            local_state_dicts.append(state_dict)
            local_train_losses.append(train_loss)
            local_val_losses.append(val_loss)
        
        # Aggregate weights
        new_state_dict = federated_averaging(local_state_dicts, sample_counts)
        global_model.load_state_dict(new_state_dict)
        
        # Calculate Weighted Metrics
        total_samples = sum(sample_counts)
        weighted_train_loss = sum(l * (c / total_samples) for l, c in zip(local_train_losses, sample_counts))
        weighted_val_loss = sum(l * (c / total_samples) for l, c in zip(local_val_losses, sample_counts))
        
        # Evaluation
        recons_df = get_reconstruction_error(global_model, x_test_loader, y_test, device)
        
        # Threshold tuning using your existing utility
        thresh_df = utils.thresholdTuning(recons_df, 10)
        TEST_THRESHOLD = thresh_df.loc[thresh_df['f1_score'].idxmax(), 'threshold']
        
        recons_df['y_pred'] = (recons_df['error'] > TEST_THRESHOLD).astype(int)
        
        fpr, tpr, _ = roc_curve(recons_df['y_true'], recons_df['error'])
        roc_auc = auc(fpr, tpr)
        
        # Metrics
        f1 = f1_score(recons_df['y_true'], recons_df['y_pred'])
        acc = accuracy_score(recons_df['y_true'], recons_df['y_pred'])

        print(f"AUC_ROC: {roc_auc*100:.4f} | F1-Score: {f1*100:.2f}% | Threshold: {TEST_THRESHOLD:.6f}")

        wandb.log({
            "round": rnd,
            "weighted_train_loss": weighted_train_loss,
            "weighted_val_loss": weighted_val_loss,
            "AUC_test": roc_auc*100,
            "F1score": f1*100
        })
        
    return global_model