import argparse
import pandas as pd
import numpy as np
import cudf as cdf
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import wandb
from sklearn import metrics

# Custom modules
import utils
import FL_DASVDD 
import autoencoder

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nbclients', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='CICIDS2017', choices=['5G3E', 'CICIDS2017'])
    parser.add_argument('--model', type=str, default='FL-DASVDD', choices=['autoencoder', 'FL-DASVDD'])
    parser.add_argument('--deactRatio', type=float, default=0.6)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--nbRounds", type=int, default=40)
    parser.add_argument("--localEpochs", type=int, default=5)
    parser.add_argument("--batchSize", type=int, default=1024)
    parser.add_argument('--K', type=float, default=0.8)
    parser.add_argument('--T', type=int, default=50)
    parser.add_argument('--Adagard_lr', type=float, default=0.0005)
    parser.add_argument('--Adagard_lr_decay', type=float, default=1e-3)
    parser.add_argument('--learningRate', type=float, default=1e-4)
    parser.add_argument('--weightDecay', type=float, default=1e-3)
    return parser.parse_args()

def run_evaluation(scores, y_true, model_name):
    scores_df = pd.DataFrame({'error': scores, 'y_true': y_true}).reset_index(drop=True)
    thresh_df = utils.thresholdTuning(scores_df, 10)
    best_thresh = thresh_df.loc[thresh_df['f1_score'].idxmax(), 'threshold']
    
    scores_df['y_pred'] = (scores_df['error'] > best_thresh).astype(int)
    
    acc = metrics.accuracy_score(y_true, scores_df['y_pred'])
    f1 = metrics.f1_score(y_true, scores_df['y_pred'])
    precision = metrics.precision_score(y_true, scores_df['y_pred'])
    recall = metrics.recall_score(y_true, scores_df['y_pred'])
    
    fpr, tpr, _ = metrics.roc_curve(y_true, scores_df['error'])
    auc_roc = metrics.auc(fpr, tpr)

    print(f"\n--- {model_name} Test Results ---")
    print(f"Threshold: {best_thresh:.6f} | AUC: {auc_roc:.4f} | F1: {f1*100:.2f}%")
    print(f"Accuracy: {acc*100:.2f}% | Precision: {precision*100:.2f}% | Recall: {recall*100:.2f}%")

if __name__ == '__main__':
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    seed = 12
    utils.set_random_seeds(seed)

    # Dataset Config Mapping
    ds_config = {
        "5G3E": {
            "dir": "FLSplit_5G3E", "test_file": "5G3E_test.csv", "val_file": "5G3E_validation.csv",
            "in_shape": 20, "code_size": 10,
            "dasvdd_model": FL_DASVDD.AE_model_5G3E, "ae_model": autoencoder.Autoencoder_5G3E
        },
        "CICIDS2017": {
            "dir": "FLSplit_CICIDS2017", "test_file": "CICIDS_test.csv", "val_file": "CICIDS_validation.csv",
            "in_shape": 11, "code_size": 8,
            "dasvdd_model": FL_DASVDD.AE_model_CICIDS2017, "ae_model": autoencoder.Autoencoder_CICIDS2017
        }
    }
    cfg = ds_config[args.dataset]
    
    # Data Loading
    test_path = f"./{cfg['dir']}/{cfg['test_file']}"
    val_path = f"./{cfg['dir']}/{cfg['val_file']}"
    
    x_test_raw = cdf.read_csv(test_path).to_pandas()
    y_test = x_test_raw['label'].to_numpy().astype(int)
    x_test_df = x_test_raw.drop(columns='label')
    x_val_df = cdf.read_csv(val_path).to_pandas()

    x_test_loader = DataLoader(TensorDataset(torch.tensor(x_test_df.to_numpy(), dtype=torch.float32)), 
                               batch_size=args.batchSize, shuffle=False, pin_memory=True)
    x_val_loader = DataLoader(TensorDataset(torch.tensor(x_val_df.to_numpy(), dtype=torch.float32)), 
                              batch_size=args.batchSize, shuffle=True, pin_memory=True)

    # Setup Paths
    split_type = f"brokerBased-{int(args.deactRatio * 100)}"
    split_type="Conv"
    model_filename = f"{args.model}-global_model_{split_type}-seed{seed}-localep{args.localEpochs}-batch{args.batchSize}.pt"
    save_path = f"./{split_type}_split/{model_filename}"

    if split_type != "Conv":
        full_path = os.path.join(cfg['dir'], f"seed_{seed}_results")
    else:
        full_path = cfg['dir']

    os.makedirs(full_path, exist_ok=True)
    os.chdir(full_path)

    print("Current working directory:", os.getcwd())
    if args.mode == "train":
        wandb.init(project=f"CLINFL-ProjectTest-{args.dataset}", 
                   name=f"{args.model}-{split_type}--seed={seed}", 
                   config={"batch_size": args.batchSize, "lr": args.learningRate})
        
        client_files = [f'./{split_type}_split/AIF_{i}_{split_type}.csv' for i in range(args.nbclients)]
        print("Client files:", client_files)

        if args.model == "FL-DASVDD":
            print("Starting FL-DASVDD training...")
            print("Startig time: ", utils.get_current_time())
            global_model, global_C, _ , weighted_gamma= FL_DASVDD.federated_train(
                device=device, dataset=args.dataset, client_files=client_files, val_loader=x_val_loader,
                test_loader=x_test_loader, y_test=y_test, in_shape=cfg['in_shape'], code_size=cfg['code_size'],
                lr=args.learningRate, batch_size=args.batchSize, global_rounds=args.nbRounds,
                localEpochs=args.localEpochs, Adagard_lr=args.Adagard_lr, Adagard_lr_decay=args.Adagard_lr_decay,
                weight_decay=args.weightDecay, K=args.K, seed=seed, T=args.T, verbose=True
            )
            print("Ending time: ", utils.get_current_time())
            torch.save({"model_state": global_model.state_dict(), "center": global_C, "global_gamma": weighted_gamma, "seed": seed}, save_path)
        
        else: # autoencoder
            global_model = autoencoder.federated_learning(seed, args.dataset, args.localEpochs, args.batchSize, args.learningRate, args.nbclients, 
                                                           args.nbRounds, split_type, x_val_loader, x_test_loader, y_test)
            torch.save(global_model.state_dict(), save_path)

        wandb.save(save_path)
        print(f"Training finished. Model saved to {save_path}")

    elif args.mode == "test":
        if args.model == "FL-DASVDD":
            checkpoint = torch.load(save_path)
            global_model = cfg['dasvdd_model'](input_shape=cfg['in_shape'], code_size=cfg['code_size']).to(device)
            global_model.load_state_dict(checkpoint["model_state"])
            
            auc, ap, scores = FL_DASVDD.DASVDD_test(global_model, checkpoint["center"], cfg['in_shape'], checkpoint["global_gamma"], 
                                                   x_test_loader, y_test, nn.MSELoss(), device)
            run_evaluation(scores, y_test, "FL-DASVDD")
            
        else: # autoencoder
            global_model = cfg['ae_model'](cfg['in_shape']).to(device)
            global_model.load_state_dict(torch.load(save_path))
            recons_df = autoencoder.get_reconstruction_error(global_model, x_test_loader, y_test, device)
            run_evaluation(recons_df['error'], y_test, "Autoencoder")