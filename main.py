import argparse
import utils
import pandas as pd
import numpy as np
import cudf as cdf
from sklearn.cluster import KMeans
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import wandb
from sklearn import metrics
import torch.nn as nn
import FL_DASVDD 

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--nbclients',type=int,default=5,help="Number of clients")
    parser.add_argument('--nbBrokers',type=int,default=3,help="Number of brokers")
    parser.add_argument('--nbDSPerBroker',type=int,default=2,help="Number of datasources per broker")
    parser.add_argument('--dataset',type=str,default='5G3E',help="dataset used for FL simulation")
    parser.add_argument('--model',type=str,default='DASVDD',help="model used for FL simulation: autoencoder or DASVDD")
    parser.add_argument('--threshold',type=float,default=0.09,help="Variance threshold for feature selection")
    parser.add_argument('--deactRatio',type=float,default=0.2,help="Ratio of deactivated links per broker")
    parser.add_argument('--generateConvFL',action='store_true',help="Whether to generate conventional FL split or broker-based split")
    parser.add_argument('--device',type=str,default='cuda:0',help="Device to use for computations")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Choose 'train' or 'test'")
    parser.add_argument("--nbRounds", type=int, default=3, help="Number of FL rounds")
    parser.add_argument("--localEpochs", type=int, default=1, help="Number of local epochs per client")
    parser.add_argument("--batchSize", type=int, default=128, help="Batch size for training")
    parser.add_argument('--K',type=float,default=0.8,help="Portion of samples used for model update in DASVDD")
    parser.add_argument('--T',type=int,default=5,help="Total number of initializations")
    parser.add_argument('--Adagard_lr',type=float,default=0.0005,help="learning rate for Adagard optimizer")
    parser.add_argument('--Adagard_lr_decay',type=float,default=0.00001,help="learning rate decay for Adagard optimizer")
    parser.add_argument('--learningRate',type=float,default=0.001,help="learning rate for local training")
    parser.add_argument('--weightDecay',type=float,default=1e-5,help="weight decay for local training")
    args=parser.parse_args()
    return args


if __name__=='__main__':
    args=get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    seed = 12
    utils.set_random_seeds(seed)
    
    if args.dataset == "5G3E":
        dir_5G3E = "FLSplit_5G3E"
        # --- Your data loading / preprocessing functions (unchanged) ---
        _, df_test_preprocessed, y_test = utils.load_and_preprocess_5G3E_data()

        x_test_df = utils.apply_pca_with_cuml(df_test_preprocessed, n_components=20)
        x_val_df=cdf.read_csv(f"./{dir_5G3E}/5G3E_validation.csv").to_pandas()

        X_test = torch.tensor(x_test_df.to_numpy(), dtype=torch.float32)
        X_val = torch.tensor(x_val_df.to_numpy(), dtype=torch.float32)

        x_val_loader = DataLoader(TensorDataset(X_val), batch_size=args.batchSize, shuffle=True, pin_memory=utils.set_pin_memory_for_device(device))
        x_test_loader = DataLoader(TensorDataset(X_test), batch_size=args.batchSize, shuffle=False, pin_memory=utils.set_pin_memory_for_device(device))
        print("X_val shape:", X_val.shape)
        print("X_test shape:", X_test.shape)

        in_shape = 20
        code_size = 10

        split_type = f"brokerBased-{int(args.deactRatio * 100)}"
        # split_type="Conv"

        if split_type != "Conv":
            out_dir = f"seed_{seed}_results"
            full_path = os.path.join(dir_5G3E, out_dir)
            os.makedirs(full_path, exist_ok=True)
            os.chdir(full_path)

        if args.mode == "train":
            client_files = [f'./{split_type}_split/AIF_{i}_{split_type}.csv' for i in range(args.nbclients)]

            # print shapes for sanity
            print("\nClients data shapes:")
            for p in client_files:
                print(cdf.read_csv(p).to_pandas().shape)
            print("-----")

            wandb.init(
                project=f"Test-{args.dataset}",
                name=f"{args.model}-{split_type}-seed{seed}-localep{args.localEpochs}-batch{args.batchSize}",
                config={
                    "batch_size": args.batchSize,
                    "learning_rate": args.learningRate,
                    "ADAGRAD LR": args.Adagard_lr,
                    "Adagrad_lr_decay": args.Adagard_lr_decay,
                    "K": args.K,
                    "T":args.T,
                    "code_size":code_size,
                    "weight_decay":args.weightDecay,
                },
            )

            if args.model != "DASVDD":
                global_model, global_C, history = FL_DASVDD.federated_train(
                    device=device,
                    client_files=client_files,
                    val_loader=x_val_loader,
                    test_loader=x_test_loader,
                    y_test=y_test,
                    in_shape=in_shape,
                    code_size=code_size,
                    lr=args.learningRate,
                    batch_size=args.batchSize,
                    global_rounds=args.nbRounds,
                    localEpochs=args.localEpochs,
                    Adagard_lr=args.Adagard_lr,
                    Adagard_lr_decay=args.Adagard_lr_decay,
                    weight_decay=args.weightDecay,
                    K=args.K,
                    seed=seed,
                    T=args.T,
                    verbose=True
                )
                # ------------------------
                # SAVE MODEL + CENTER
                # ------------------------
                torch.save({
                    "model_state": global_model.state_dict(),
                    "center": global_C,
                    "seed": seed
                }, f"./{split_type}_split/{args.model}-global_model_{split_type}-seed{seed}-localep{args.localEpochs}-batch{args.batchSize}.pt")

                wandb.save(f"./{split_type}_split/{args.model}-global_model_{split_type}-seed{seed}-localep{args.localEpochs}-batch{args.batchSize}.pt")

                print("Training finished. Final global center norm:", global_C.norm().item())
            
            elif args.model == "autoencoder":
                pass  # Implement autoencoder training if needed

        elif args.mode == "test":
            ####################################################################################
            # Test after training

            if args.model != "DASVDD":
                model_path = f"{split_type}_split/{args.model}-global_model_{split_type}-seed{seed}-localep{args.localEpochs}-batch{args.batchSize}.pt"

                # Load the checkpoint
                checkpoint = torch.load(model_path)

                # Extract the components
                model_state = checkpoint["model_state"]
                global_C = checkpoint["center"]
                seed = checkpoint["seed"]

                global_model = FL_DASVDD.AE_model_5G3E(input_shape=in_shape, code_size=code_size).to(device)
                # Load state into your model
                global_model.load_state_dict(model_state)
                auc, ap, scores = FL_DASVDD.DASVDD_test(global_model, global_C, in_shape, 0.0, x_test_loader, y_test, nn.MSELoss(), device)
                print(f"global_C norm: {global_C.norm().item():.6f}")
                print(f"Final Test AUC: {auc:.2f}, AP: {ap:.2f}")

                # here error is the score
                scores_df = pd.DataFrame({'error': scores, 'y_true': y_test}).reset_index(drop=True)
                thresh_df = utils.thresholdTuning(scores_df, 10)
                TEST_THRESHOLD = thresh_df[thresh_df['f1_score'] == thresh_df['f1_score'].max()]['threshold'].values[0]

                print("\nResults:")
                print(f"Optimal Threshold: {TEST_THRESHOLD:.6f}")

                scores_df['y_pred'] = scores_df['error'] > TEST_THRESHOLD

                accuracy = metrics.accuracy_score(scores_df['y_true'], scores_df['y_pred']) 
                precision = metrics.precision_score(scores_df['y_true'], scores_df['y_pred'])
                recall = metrics.recall_score(scores_df['y_true'], scores_df['y_pred'])
                f1 = metrics.f1_score(scores_df['y_true'], scores_df['y_pred'])
                mcc = metrics.matthews_corrcoef(scores_df['y_true'], scores_df['y_pred'])

                print(f"Accuracy: {accuracy*100:.3f}%)")
                print(f"Precision: {precision*100:.3f}%)")
                print(f"Recall: {recall*100:.3f}%)")
                print(f"F1 Score: {f1*100:.3f}/%)")
                print(f"MCC: {mcc*100:.3f}/%)")