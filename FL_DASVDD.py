import os
import sys
import math
import copy
import faulthandler
log = open("crash.log", "w")
faulthandler.enable(file=log)
import collections
import numpy as np
import pandas as pd
import cudf
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn import metrics
import wandb
import utils


class AE_model_5G3E(nn.Module):
    def __init__(self, input_shape, code_size,**kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 25),
            nn.LeakyReLU(),
            nn.Linear(25, code_size),
            nn.LeakyReLU(),
            nn.Linear(code_size, code_size),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_size, code_size),
            nn.LeakyReLU(),
            nn.Linear(code_size, 25),
            nn.LeakyReLU(),
            nn.Linear(25, input_shape)
        )

    def forward(self, x):
        code = self.encoder(x)
        recon = self.decoder(code)
        return recon, code

class AE_model_CICIDS2017(nn.Module):
    def __init__(self, input_shape, code_size,**kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 15),
            nn.LeakyReLU(),
            nn.Linear(15, code_size),
            nn.LeakyReLU(),
            nn.Linear(code_size, code_size),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_size, code_size),
            nn.LeakyReLU(),
            nn.Linear(code_size, 15),
            nn.LeakyReLU(),
            nn.Linear(15, input_shape)
        )

    def forward(self, x):
        code = self.encoder(x)
        recon = self.decoder(code)
        return recon, code
    
# FedAvg weighted aggregator
def fedavg(client_states, client_ns):
    if len(client_states) == 0:
        raise ValueError("No client states to aggregate.")
    total = sum(client_ns)
    if total == 0:
        raise ValueError("Sum of client sample counts is zero.")
    global_state = collections.OrderedDict()
    for k in client_states[0].keys():
        global_state[k] = torch.zeros_like(client_states[0][k])
    for state, n in zip(client_states, client_ns):
        w = n / total
        for k in global_state.keys():
            global_state[k] += state[k].to(global_state[k].device) * w
    return global_state


def load_client_loader(csv_path, batch_size, in_shape=None, shuffle=True, device=torch.device("cpu")):
    df = cudf.read_csv(csv_path).to_pandas()
    X = torch.tensor(df.to_numpy(), dtype=torch.float32)
    ds = TensorDataset(X)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=utils.set_pin_memory_for_device(device))
    return loader, len(df)


# ---------------------------------------------------------------------------
# Time cost (paper Eq. 3): t_{k,m} = per-round communication time for a client
# to upload/download the model parameters. With no real network in the
# simulation we estimate it from the model size and a link bandwidth.
# Upload + download => num_transfers=2. Parameters are float32 (element_size()).
# ---------------------------------------------------------------------------
def estimate_comm_time(model, bandwidth_mbps, num_transfers=2):
    """Per-round communication time (seconds) to move the model parameters over
    a link of `bandwidth_mbps` megabits/sec, `num_transfers` times
    (default 2 = download global model + upload local model)."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    bits = num_transfers * param_bytes * 8.0
    return bits / (bandwidth_mbps * 1e6)


# ---------------------------------------------------------------------------
# Communication volume: total bytes transferred between clients and server.
# Each round every client downloads (global model + global center) and uploads
# (local model + local center). state_dict_bytes counts every tensor exactly.
# ---------------------------------------------------------------------------
def state_dict_bytes(state_dict):
    return sum(t.numel() * t.element_size() for t in state_dict.values())


def _fmt_bytes(b):
    b = float(b)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024.0 or unit == "TB":
            return f"{b:.2f} {unit}"
        b /= 1024.0


@torch.no_grad()
def tune_gamma(AE_class, in_shape, code_size, train_loader, device, T=10, eps=1e-8):

    device = torch.device(device)
    accum = 0.0
    used = 0
    for t in range(T):
        model = AE_class(input_shape=in_shape,code_size=code_size).to(device)
        model.eval()
        RE_sum = 0.0
        R_sum = 0.0
        n_total = 0
        for batch in train_loader:
            if isinstance(batch, list):
                batch = batch[0]
            x = batch.view(-1, in_shape).to(device)
            recon, code = model(x)
            # per-sample reconstruction (mean squared error per sample)
            per_sample_re = torch.mean((recon - x) ** 2, dim=1) 
            per_sample_r = torch.sum(code ** 2, dim=1)        
            RE_sum += float(per_sample_re.sum().item())
            R_sum += float(per_sample_r.sum().item())
            n_total += x.size(0)
        if n_total == 0:
            continue
        RE_mean = RE_sum / n_total
        R_mean = R_sum / n_total
        ratio = RE_mean / (R_mean + eps)
        accum += ratio
        used += 1
    gamma = accum / max(1, used)
    return float(gamma)


def DASVDD_validate(model, in_shape, C, val_loader, criterion, Gamma, device):
    """
    Validate model and also validate the learned center C by comparing it
    to the mean of the latent codes over the entire validation set.

    Returns:
        val_loss_avg, ae_loss_avg, svdd_loss_avg, center_mse, center_l2
    """
    model.to(device)
    model.eval()

    val_loss = 0.0
    ae_loss = 0.0
    svdd_loss = 0.0
    num_batches = 0

    all_codes = []
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, list):
                batch = batch[0]
            x = batch.view(-1, in_shape).to(device)
            outputs, code = model(x)

            # per-batch (mean) AE loss and SVDD loss (mean over samples)
            batch_ae_loss = torch.mean((outputs - x) ** 2)
            batch_svdd_loss = torch.mean(torch.sum((code - C.to(device)) ** 2, dim=1))
            batch_loss = batch_ae_loss + Gamma * batch_svdd_loss

            val_loss += batch_loss.item()
            ae_loss += batch_ae_loss.item()
            svdd_loss += batch_svdd_loss.item()
            num_batches += 1

            all_codes.append(code.detach().cpu())

    # compute averages
    val_loss_avg = val_loss / num_batches
    ae_loss_avg = ae_loss / num_batches
    svdd_loss_avg = svdd_loss / num_batches

    # compute global mean code over the entire validation set
    codes_cat = torch.cat(all_codes, dim=0)      
    mean_code = codes_cat.mean(dim=0).to(C.device) 
    center_mse = float(criterion(C, mean_code).item())

    return val_loss_avg, ae_loss_avg, svdd_loss_avg, center_mse

def DASVDD_test(model,C,in_shape,Gamma,test_loader,labels,criterion,device):

  with torch.no_grad():
    score = []
    for x_test in test_loader:
      if isinstance(x_test, list):
          x_test = x_test[0]
      x_test =x_test.view(-1, in_shape).to(device)
      x_test_hat,code_test = model(x_test)
      loss = criterion(x_test_hat,x_test) + Gamma*torch.sum((code_test.to(device) - C) ** 2, dim=1)
      score.extend(loss.cpu().numpy().tolist())
    score = np.array(score)
  scores = np.array(score)
  labels = np.array(labels)


  return metrics.roc_auc_score(labels,score)*100, metrics.average_precision_score(labels,score)*100,scores

def client_local_update(model, in_shape, client_loader, val_loader,
                        client_n_samples, device, optimizer,scheduler, update_center,
                        K, localEpochs, C_local, global_gamma,criterion ):
    device = torch.device(device)
    model.to(device)
    C_local = C_local.to(device)
    C_local.requires_grad = True
    model.train()
    last_R = 0.0

    batch_count = 0
    total_loss = 0.0
    total_ae = 0.0
    total_svdd = 0.0
    total_center_loss=0.0

    # ---- Time cost (paper Eq. 3): measure local computation time only ----
    # We time the local training loop (all E = localEpochs iterations). CUDA is
    # asynchronous, so we synchronize before/after to get an accurate wall time.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    compute_start = time.perf_counter()

    for epoch in range(max(1, int(localEpochs))):
        for batch in client_loader:
            if isinstance(batch, list):
                batch = batch[0]
            x = batch.view(-1, in_shape).to(device)
            N = x.size(0)
            if N == 0:
                continue
            Num_batch = int(math.ceil(K * N))
            Num_batch = max(1, Num_batch)

            optimizer.zero_grad()
            outputs, code = model(x[:Num_batch, :])
            R = torch.mean(torch.sum((code - C_local) ** 2, dim=1))
            train_loss = criterion(outputs, x[:Num_batch, :]) + global_gamma * R
            train_loss.backward()
            optimizer.step()

            total_loss += train_loss.item()
            total_ae += criterion(outputs, x[:Num_batch, :]).item()
            total_svdd += R.item()
            last_R = R.item()
            batch_count += 1

            # center update using leftover samples
            if Num_batch < N:
                _, c_code = model(x[Num_batch:, :])
                if c_code.numel() > 0:
                    center = torch.mean(c_code, dim=0).detach()
                    update_center.zero_grad()
                    center_loss = criterion(C_local, center)
                    center_loss.backward()
                    update_center.step()
                    total_center_loss += center_loss.item()

    # ---- stop the compute timer (training only, excludes validation) ----
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    compute_time = time.perf_counter() - compute_start          # = t_{k,p} * E
    t_kp = compute_time / max(1, int(localEpochs))              # per local iteration

    if batch_count == 0:
        avg_loss = avg_ae = avg_svdd = 0.0
        avg_center_loss = 0.0
    else:
        avg_loss = total_loss / batch_count
        avg_ae = total_ae / batch_count
        avg_svdd = total_svdd / batch_count
        avg_center_loss = total_center_loss / batch_count

    # validation metrics
    val_loss, val_ae, val_svdd, center_mse = DASVDD_validate(model, in_shape, C_local.detach(), val_loader, criterion, global_gamma, device)

    # Return updated state dict, n_samples, train metrics, val metrics, client center (cpu), last_R, gamma
    # NOTE: compute_time (= t_{k,p}*E) and t_kp are returned inside train metrics
    # so the function signature stays unchanged.
    return copy.deepcopy(model.state_dict()), client_n_samples, \
           {"train_loss": avg_loss, "ae_loss": avg_ae, "svdd_loss": avg_svdd,
            "compute_time": compute_time, "t_kp": t_kp}, \
           {"val_loss": val_loss, "val_ae_loss": val_ae, "val_svdd_loss": val_svdd, "val_center_mse":center_mse}, \
           avg_center_loss,C_local.detach().cpu(), last_R, global_gamma


def federated_train(device, dataset,client_files, val_loader, test_loader, y_test,
                    in_shape, code_size, lr=1e-3, batch_size=256, global_rounds=100,
                    localEpochs=1, Adagard_lr=0.01, Adagard_lr_decay=0.0, weight_decay=1e-2,K=0.8,
                    client_comm_times=None, comm_bandwidth_mbps=100.0,
                    seed=12, T=10, verbose=True):
    utils.set_random_seeds(seed)
    device = torch.device(device)
    early_stop_patience = 100   # number of rounds to wait for improvement
    best_val_loss = float("inf")
    best_center_mse = float("inf")
    rounds_no_improve_loss = 0
    rounds_no_improve_center = 0
    nb_clients = len(client_files)
    client_loaders = []
    client_counts = []

    # ---- Time cost (paper Eq. 5): total learning time accumulator t_tot ----
    total_time = 0.0

    if dataset == "5G3E":
        model_class = AE_model_5G3E
    elif dataset == "CICIDS2017":
        model_class = AE_model_CICIDS2017
    for path in client_files:
        loader, n = load_client_loader(path, batch_size=batch_size, in_shape=in_shape, shuffle=True, device=device)
        client_loaders.append(loader)
        client_counts.append(n)

    # compute gamma per client and weighted aggregate
    if verbose:
        print("Computing local gamma for each client (T={}):".format(T))
    client_gammas = []
    for i, loader in enumerate(client_loaders):
        g = tune_gamma(model_class, in_shape, code_size, loader, device, T=T)
        client_gammas.append(g)
        if verbose:
            print(f" Client {i}: gamma={g:.6f}")

    total_samples = sum(client_counts)
    if total_samples == 0:
        raise RuntimeError("No samples across clients.")
    weighted_gamma = sum(g * n for g, n in zip(client_gammas, client_counts)) / float(total_samples)
    if verbose:
        print("Initial aggregated global gamma:", weighted_gamma)

    # initialize global model and center
    global_model = model_class(input_shape=in_shape, code_size=code_size).to(device)
    global_state = global_model.state_dict()
    global_C = torch.randn(code_size, device=device, requires_grad=False)
    criterion = nn.MSELoss()

    # ---- Time cost (paper Eq. 3): per-client communication time t_{k,m} ----
    # If not supplied, estimate it from the model size and a link bandwidth
    # (same value for every client). Pass `client_comm_times` (one entry per
    # client) to model heterogeneous links / stragglers as in Fig. 1.
    if client_comm_times is None:
        per_client_comm = estimate_comm_time(global_model, comm_bandwidth_mbps, num_transfers=2)
        client_comm_times = [per_client_comm] * nb_clients
    else:
        if len(client_comm_times) != nb_clients:
            raise ValueError("client_comm_times must have one entry per client.")
        client_comm_times = [float(c) for c in client_comm_times]
    if verbose:
        print("Per-client communication time t_{k,m} (s):",
              [round(c, 6) for c in client_comm_times])

    # ---- Communication volume: bytes exchanged with the server each round ----
    # Per client: download (global model + global center) + upload (local model
    # + local center). Sizes are constant across rounds for this architecture.
    model_state_bytes = state_dict_bytes(global_state)
    center_bytes = global_C.numel() * global_C.element_size()
    bytes_per_client_round = 2 * (model_state_bytes + center_bytes)   # down + up
    total_bytes = 0          # total transferred (clients <-> server)
    total_bytes_down = 0     # server -> clients
    total_bytes_up = 0       # clients -> server
    if verbose:
        print(f"Payload per client/round: model={_fmt_bytes(model_state_bytes)} "
              f"+ center={_fmt_bytes(center_bytes)} -> "
              f"{_fmt_bytes(bytes_per_client_round)} (download + upload)")

    # federated rounds
    history = []
    for rnd in range(1, int(global_rounds) + 1):
        if verbose:
            print(f"\n--- Round {rnd}/{global_rounds} ---")

        client_states = []
        client_ns = []
        client_Cs = []
        client_Rs = []
        client_train_metrics_list = []
        client_val_metrics_list = []
        client_center_loss_list = []
        client_compute_times = []   # t_{k,p} * E for each client this round

        for cid in range(nb_clients):
            # local model: copy global weights and move to device
            local_model = model_class(input_shape=in_shape,code_size=code_size)
            local_model.load_state_dict(copy.deepcopy(global_state))
            local_model.to(device)

            # create local center from global center 
            local_C = global_C.clone().detach().to(device)
            local_C.requires_grad = True

            # optimizers after model on device
            local_optimizer = torch.optim.Adam(local_model.parameters(), lr=lr,weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(local_optimizer, gamma=0.99) 
            local_update_center = torch.optim.Adagrad([local_C], lr=Adagard_lr, lr_decay=Adagard_lr_decay)

            # run local update
            state_dict, n, train_metrics, val_metrics, avg_center_loss,C_client, R_client, local_gamma = client_local_update(
                model=local_model,
                in_shape=in_shape,
                client_loader=client_loaders[cid],
                val_loader=val_loader,
                client_n_samples=client_counts[cid],
                device=device,
                optimizer=local_optimizer,
                scheduler=scheduler,
                update_center=local_update_center,
                K=K,
                localEpochs=localEpochs,
                C_local=local_C,
                global_gamma=weighted_gamma,
                criterion=criterion
            )

            client_states.append(state_dict)
            client_ns.append(n)
            client_Cs.append(C_client) 
            client_Rs.append(R_client)
            client_train_metrics_list.append(train_metrics)
            client_val_metrics_list.append(val_metrics)
            client_center_loss_list.append(avg_center_loss)
            client_compute_times.append(train_metrics["compute_time"])  # = t_{k,p}*E

            if verbose:
                print(f" client {cid}: n={n}, train_loss={train_metrics['train_loss']:.4f}, "
                      f"R={R_client:.6f}, C_norm={C_client.norm().item():.6f}, "
                      f"t_compute={train_metrics['compute_time']:.4f}s")

            # free local model & optimizer memory
            del local_model, local_optimizer, local_update_center
            torch.cuda.empty_cache()

        # ---- Time cost (paper Eqs. 3-5) ----
        # Eq. (3): t_k = t_{k,p} * E + t_{k,m}; compute_time already equals t_{k,p}*E
        client_t_k = [ct + cm for ct, cm in zip(client_compute_times, client_comm_times)]
        # Eq. (4): per-round time is set by the slowest participating client (straggler)
        straggler = int(np.argmax(client_t_k))
        round_time = float(client_t_k[straggler])
        # Eq. (5): total learning time after R rounds
        total_time += round_time

        # ---- Communication volume this round (bytes) ----
        round_bytes_down = nb_clients * (model_state_bytes + center_bytes)
        round_bytes_up = nb_clients * (model_state_bytes + center_bytes)
        round_bytes = round_bytes_down + round_bytes_up
        total_bytes_down += round_bytes_down
        total_bytes_up += round_bytes_up
        total_bytes += round_bytes

        # Aggregate models (FedAvg)
        global_state = fedavg(client_states, client_ns)
        global_model.load_state_dict(global_state)

        # Aggregate centers by weighted average
        total = sum(client_ns)
        weighted_C = torch.zeros_like(client_Cs[0])
        for c, w in zip(client_Cs, client_ns):
            weighted_C += c * (w / total)
        global_C = weighted_C.to(device)
        global_C.requires_grad = False

        # weighted metrics
        weighted_train_loss = sum(m['train_loss'] * n for m, n in zip(client_train_metrics_list, client_ns)) / total
        weighted_train_ae = sum(m['ae_loss'] * n for m, n in zip(client_train_metrics_list, client_ns)) / total
        weighted_train_svdd = sum(m['svdd_loss'] * n for m, n in zip(client_train_metrics_list, client_ns)) / total
        weighted_center_loss = sum(cl * n for cl, n in zip(client_center_loss_list, client_ns)) / sum(client_ns)


        weighted_val_loss = sum(m['val_loss'] * n for m, n in zip(client_val_metrics_list, client_ns)) / total
        weighted_val_ae = sum(m['val_ae_loss'] * n for m, n in zip(client_val_metrics_list, client_ns)) / total
        weighted_val_svdd = sum(m['val_svdd_loss'] * n for m, n in zip(client_val_metrics_list, client_ns)) / total
        weighted_val_center_mse = sum(m['val_center_mse'] * n for m, n in zip(client_val_metrics_list, client_ns)) / total

        weighted_R = sum(r * n for r, n in zip(client_Rs, client_ns)) / total

        # Check for improvement on weighted validation loss
        if weighted_val_loss  < best_val_loss:
            best_val_loss = weighted_val_loss
            rounds_no_improve_loss = 0
        else:
            rounds_no_improve_loss += 1

        # Check for improvement on weighted validation center MSE
        if weighted_val_center_mse  < best_center_mse:
            best_center_mse = weighted_val_center_mse
            rounds_no_improve_center = 0
        else:
            rounds_no_improve_center += 1

        # Trigger early stopping if EITHER metric hasn't improved for `early_stop_patience` rounds
        if rounds_no_improve_loss >= early_stop_patience or rounds_no_improve_center >= early_stop_patience:
            if verbose:
                reason = []
                if rounds_no_improve_loss >= early_stop_patience:
                    reason.append(f"val_loss no improvement for {early_stop_patience} rounds")
                if rounds_no_improve_center >= early_stop_patience:
                    reason.append(f"val_center_mse no improvement for {early_stop_patience} rounds")
                print(f"Early stopping triggered at round {rnd} ({' and '.join(reason)})")
                print(f"Total learning time t_tot so far (Eq. 5): {total_time:.4f} s")
                print(f"Total bytes transferred so far: {_fmt_bytes(total_bytes)} "
                      f"(down {_fmt_bytes(total_bytes_down)} / up {_fmt_bytes(total_bytes_up)})")
            break

        # Evaluate on test set
        try:
            auc, ap ,_= DASVDD_test(global_model, global_C, in_shape, weighted_gamma, test_loader, y_test,criterion, device)
        except Exception as e:
            auc, ap = float('nan'), float('nan')
            if verbose:
                print("Warning: error in DASVDD_test:", e)

        # wandb logging
        wandb.log({
            "round": rnd,
            "weighted_train_loss": weighted_train_loss,
           # "weighted_train_ae": weighted_train_ae,
            #"weighted_train_svdd": weighted_train_svdd,
            "weighted_center_loss": weighted_center_loss,
            "weighted_val_loss": weighted_val_loss,
           # "weighted_val_ae": weighted_val_ae,
            #"weighted_val_svdd": weighted_val_svdd,
           # "weighted_val_center_mse":weighted_val_center_mse,
            "global_R": weighted_R,
            "global_gamma": weighted_gamma,
           # "global_center_norm": global_C.norm().item(),
           # "global_C-mean": global_C.mean().item(),    
           # "global_C[0]":global_C[0],  
            "AUC_test": auc,
            "AP_test": ap,
            # ---- Time cost (paper Eqs. 3-5) ----
            "round_time_sec": round_time,            # t^(r) = max_k t_k   (Eq. 4)
            "total_time_sec": total_time,            # t_tot = sum_r t^(r) (Eq. 5)
            "straggler_client": straggler,           # slowest client this round
            "max_compute_time_sec": max(client_compute_times),
            "comm_time_sec": client_comm_times[straggler],
            # ---- Communication volume (bytes transferred) ----
            "round_bytes": round_bytes,              # this round, both directions
            "total_bytes": total_bytes,              # cumulative, both directions
            "total_bytes_down": total_bytes_down,    # cumulative server -> clients
            "total_bytes_up": total_bytes_up,        # cumulative clients -> server
            "total_MB": total_bytes / (1024.0 ** 2)
        })

        if verbose:
            print(f"Round {rnd} — AUC: {auc:.2f}, weighted_val_loss: {weighted_val_loss:.6f}, "
                  f"global_C_norm: {global_C.norm().item():.6f}")
            print(f"           round_time t^(r)={round_time:.4f}s "
                  f"(straggler=client {straggler}), total_time t_tot={total_time:.4f}s")
            print(f"           round_bytes={_fmt_bytes(round_bytes)}, "
                  f"total_transferred={_fmt_bytes(total_bytes)}")

        # store history
        history.append({
            "round": rnd,
            "auc": auc, "ap": ap,
            "weighted_val_loss": weighted_val_loss,
            "global_C_norm": global_C.norm().item(),
            # ---- Time cost (paper Eqs. 4-5) ----
            "round_time_sec": round_time,
            "total_time_sec": total_time,
            "straggler_client": straggler,
            # ---- Communication volume (bytes) ----
            "round_bytes": round_bytes,
            "total_bytes": total_bytes,
            "total_bytes_down": total_bytes_down,
            "total_bytes_up": total_bytes_up
        })

    if verbose:
        print(f"\n=== Total learning time t_tot(K, E={localEpochs}, R) (Eq. 5): "
              f"{total_time:.4f} s over {rnd} round(s) ===")
        print(f"=== Total bytes transferred (clients <-> server): {_fmt_bytes(total_bytes)} "
              f"(down {_fmt_bytes(total_bytes_down)} / up {_fmt_bytes(total_bytes_up)}) ===")

    return global_model, global_C, history, weighted_gamma
