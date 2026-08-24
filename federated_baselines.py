import copy
import math
import time
import collections

import numpy as np
import torch
import torch.nn as nn
import wandb

import utils
import FL_DASVDD as dasvdd

# ---------------------------------------------------------------------------
# FedProx and SCAFFOLD baselines for the FL-DASVDD framework: same
# architectures (AE_model_5G3E / AE_model_CICIDS2017), same reconstruction +
# SVDD objective, same learned center C, same gamma tuning and DASVDD_test
# evaluation as FL_DASVDD.federated_train. Only the per-client local-update
# rule and the way client updates are folded into the global model differ:
#   - FedAvg (FL-DASVDD):  local Adam step on recon+SVDD loss, weighted avg
#   - FedProx:             + proximal term mu/2*||w - w_global||^2
#   - SCAFFOLD:            local SGD steps corrected by control variates
# The center C is aggregated the same (weighted average) way in all three,
# so the comparison isolates the model-parameter aggregation strategy rather
# than mixing in a different treatment of the auxiliary center variable.
# Like FL-DASVDD, none of these move raw client data — only model parameters
# (+ small control variates for SCAFFOLD) cross the client/server boundary.
# ---------------------------------------------------------------------------


def _get_model_class(dataset):
    if dataset == "CICIDS2017":
        return dasvdd.AE_model_CICIDS2017
    elif dataset == "5G3E":
        return dasvdd.AE_model_5G3E
    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# FedProx local update: identical to FL_DASVDD.client_local_update (same
# K-split between model training and center update) with an added proximal
# term that pulls the local model back towards the global model.
# ---------------------------------------------------------------------------

def dasvdd_local_update_fedprox(model, in_shape, client_loader, val_loader, client_n_samples,
                                 device, optimizer, update_center, K, localEpochs, C_local,
                                 global_gamma, criterion, global_params, mu):
    device = torch.device(device)
    model.to(device)
    C_local = C_local.to(device)
    C_local.requires_grad = True
    model.train()
    global_params = [p.detach().clone().to(device) for p in global_params]

    last_R = 0.0
    batch_count = 0
    total_loss = total_ae = total_svdd = total_center_loss = 0.0

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
            Num_batch = max(1, int(math.ceil(K * N)))

            optimizer.zero_grad()
            outputs, code = model(x[:Num_batch, :])
            R = torch.mean(torch.sum((code - C_local) ** 2, dim=1))
            ae_loss = criterion(outputs, x[:Num_batch, :])
            prox_term = sum((p - g).pow(2).sum() for p, g in zip(model.parameters(), global_params))
            train_loss = ae_loss + global_gamma * R + (mu / 2.0) * prox_term
            train_loss.backward()
            optimizer.step()

            total_loss += train_loss.item()
            total_ae += ae_loss.item()
            total_svdd += R.item()
            last_R = R.item()
            batch_count += 1

            if Num_batch < N:
                _, c_code = model(x[Num_batch:, :])
                if c_code.numel() > 0:
                    center = torch.mean(c_code, dim=0).detach()
                    update_center.zero_grad()
                    center_loss = criterion(C_local, center)
                    center_loss.backward()
                    update_center.step()
                    total_center_loss += center_loss.item()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    compute_time = time.perf_counter() - compute_start
    t_kp = compute_time / max(1, int(localEpochs))

    if batch_count == 0:
        avg_loss = avg_ae = avg_svdd = avg_center_loss = 0.0
    else:
        avg_loss = total_loss / batch_count
        avg_ae = total_ae / batch_count
        avg_svdd = total_svdd / batch_count
        avg_center_loss = total_center_loss / batch_count

    val_loss, val_ae, val_svdd, center_mse = dasvdd.DASVDD_validate(
        model, in_shape, C_local.detach(), val_loader, criterion, global_gamma, device
    )

    return copy.deepcopy(model.state_dict()), client_n_samples, \
        {"train_loss": avg_loss, "ae_loss": avg_ae, "svdd_loss": avg_svdd,
         "compute_time": compute_time, "t_kp": t_kp}, \
        {"val_loss": val_loss, "val_ae_loss": val_ae, "val_svdd_loss": val_svdd, "val_center_mse": center_mse}, \
        avg_center_loss, C_local.detach().cpu(), last_R, global_gamma


def federated_train_fedprox(device, dataset, client_files, val_loader, test_loader, y_test,
                             in_shape, code_size, lr=1e-3, batch_size=256, global_rounds=100,
                             localEpochs=1, Adagard_lr=0.01, Adagard_lr_decay=0.0, weight_decay=1e-2,
                             K=0.8, mu=0.01, client_comm_times=None, comm_bandwidth_mbps=100.0,
                             seed=12, T=10, verbose=True):
    utils.set_random_seeds(seed)
    device = torch.device(device)
    nb_clients = len(client_files)
    client_loaders = []
    client_counts = []
    total_time = 0.0

    model_class = _get_model_class(dataset)

    for path in client_files:
        loader, n = dasvdd.load_client_loader(path, batch_size=batch_size, in_shape=in_shape,
                                               shuffle=True, device=device)
        client_loaders.append(loader)
        client_counts.append(n)

    if verbose:
        print(f"[FedProx] Computing local gamma for each client (T={T}):")
    client_gammas = []
    for i, loader in enumerate(client_loaders):
        g = dasvdd.tune_gamma(model_class, in_shape, code_size, loader, device, T=T)
        client_gammas.append(g)
        if verbose:
            print(f" Client {i}: gamma={g:.6f}")

    total_samples = sum(client_counts)
    if total_samples == 0:
        raise RuntimeError("No samples across clients.")
    weighted_gamma = sum(g * n for g, n in zip(client_gammas, client_counts)) / float(total_samples)
    if verbose:
        print("Initial aggregated global gamma:", weighted_gamma)

    global_model = model_class(input_shape=in_shape, code_size=code_size).to(device)
    global_state = global_model.state_dict()
    global_C = torch.randn(code_size, device=device, requires_grad=False)
    criterion = nn.MSELoss()

    if client_comm_times is None:
        per_client_comm = dasvdd.estimate_comm_time(global_model, comm_bandwidth_mbps, num_transfers=2)
        client_comm_times = [per_client_comm] * nb_clients
    else:
        client_comm_times = [float(c) for c in client_comm_times]

    model_state_bytes = dasvdd.state_dict_bytes(global_state)
    center_bytes = global_C.numel() * global_C.element_size()
    total_bytes = total_bytes_down = total_bytes_up = 0

    history = []
    for rnd in range(1, int(global_rounds) + 1):
        if verbose:
            print(f"\n[FedProx] --- Round {rnd}/{global_rounds} ---")

        client_states, client_ns, client_Cs, client_Rs = [], [], [], []
        client_train_metrics_list, client_val_metrics_list, client_center_loss_list = [], [], []
        client_compute_times = []

        global_params = list(global_model.parameters())

        for cid in range(nb_clients):
            local_model = model_class(input_shape=in_shape, code_size=code_size)
            local_model.load_state_dict(copy.deepcopy(global_state))
            local_model.to(device)

            local_C = global_C.clone().detach().to(device)
            local_C.requires_grad = True

            local_optimizer = torch.optim.Adam(local_model.parameters(), lr=lr, weight_decay=weight_decay)
            local_update_center = torch.optim.Adagrad([local_C], lr=Adagard_lr, lr_decay=Adagard_lr_decay)

            state_dict, n, train_metrics, val_metrics, avg_center_loss, C_client, R_client, _ = \
                dasvdd_local_update_fedprox(
                    model=local_model, in_shape=in_shape, client_loader=client_loaders[cid],
                    val_loader=val_loader, client_n_samples=client_counts[cid], device=device,
                    optimizer=local_optimizer, update_center=local_update_center, K=K,
                    localEpochs=localEpochs, C_local=local_C, global_gamma=weighted_gamma,
                    criterion=criterion, global_params=global_params, mu=mu
                )

            client_states.append(state_dict)
            client_ns.append(n)
            client_Cs.append(C_client)
            client_Rs.append(R_client)
            client_train_metrics_list.append(train_metrics)
            client_val_metrics_list.append(val_metrics)
            client_center_loss_list.append(avg_center_loss)
            client_compute_times.append(train_metrics["compute_time"])

            if verbose:
                print(f" client {cid}: n={n}, train_loss={train_metrics['train_loss']:.4f}, "
                      f"R={R_client:.6f}, C_norm={C_client.norm().item():.6f}")

            del local_model, local_optimizer, local_update_center
            torch.cuda.empty_cache()

        client_t_k = [ct + cm for ct, cm in zip(client_compute_times, client_comm_times)]
        straggler = int(np.argmax(client_t_k))
        round_time = float(client_t_k[straggler])
        total_time += round_time

        round_bytes = 2 * nb_clients * (model_state_bytes + center_bytes)
        total_bytes_down += nb_clients * (model_state_bytes + center_bytes)
        total_bytes_up += nb_clients * (model_state_bytes + center_bytes)
        total_bytes += round_bytes

        global_state = dasvdd.fedavg(client_states, client_ns)
        global_model.load_state_dict(global_state)

        total = sum(client_ns)
        weighted_C = torch.zeros_like(client_Cs[0])
        for c, w in zip(client_Cs, client_ns):
            weighted_C += c * (w / total)
        global_C = weighted_C.to(device)
        global_C.requires_grad = False

        weighted_train_loss = sum(m['train_loss'] * n for m, n in zip(client_train_metrics_list, client_ns)) / total
        weighted_center_loss = sum(cl * n for cl, n in zip(client_center_loss_list, client_ns)) / total
        weighted_val_loss = sum(m['val_loss'] * n for m, n in zip(client_val_metrics_list, client_ns)) / total
        weighted_val_center_mse = sum(m['val_center_mse'] * n for m, n in zip(client_val_metrics_list, client_ns)) / total
        weighted_R = sum(r * n for r, n in zip(client_Rs, client_ns)) / total

        try:
            auc, ap, _ = dasvdd.DASVDD_test(global_model, global_C, in_shape, weighted_gamma,
                                             test_loader, y_test, criterion, device)
        except Exception as e:
            auc, ap = float('nan'), float('nan')
            if verbose:
                print("Warning: error in DASVDD_test:", e)

        wandb.log({
            "round": rnd,
            "weighted_train_loss": weighted_train_loss,
            "weighted_center_loss": weighted_center_loss,
            "weighted_val_loss": weighted_val_loss,
            "global_R": weighted_R,
            "global_gamma": weighted_gamma,
            "AUC_test": auc,
            "AP_test": ap,
            "fedprox_mu": mu,
            "round_time_sec": round_time,
            "total_time_sec": total_time,
            "straggler_client": straggler,
            "round_bytes": round_bytes,
            "total_bytes": total_bytes,
            "total_bytes_down": total_bytes_down,
            "total_bytes_up": total_bytes_up,
        })

        if verbose:
            print(f"[FedProx] Round {rnd} — AUC: {auc:.2f}, weighted_val_loss: {weighted_val_loss:.6f}, "
                  f"total_time={total_time:.4f}s")

        history.append({"round": rnd, "auc": auc, "ap": ap, "weighted_val_loss": weighted_val_loss,
                         "round_time_sec": round_time, "total_time_sec": total_time})

    return global_model, global_C, history, weighted_gamma


# ---------------------------------------------------------------------------
# SCAFFOLD local update: same K-split DASVDD objective, but the model-weight
# gradient is corrected with control variates (g - c_i + c) and applied via
# a plain SGD step (per the original algorithm — no adaptive optimizer),
# while the center C keeps the same uncorrected Adagrad update as vanilla
# FL-DASVDD/FedProx.
# ---------------------------------------------------------------------------

def dasvdd_local_update_scaffold(model, in_shape, client_loader, val_loader, client_n_samples,
                                  device, lr, update_center, K, localEpochs, C_local, global_gamma,
                                  criterion, c_global, c_local, grad_clip_norm=5.0):
    device = torch.device(device)
    model.to(device)
    C_local = C_local.to(device)
    C_local.requires_grad = True
    model.train()

    x_global = [p.detach().clone().to(device) for p in model.parameters()]
    c_global = [c.to(device) for c in c_global]
    c_local = [c.to(device) for c in c_local]

    last_R = 0.0
    batch_count = 0
    num_steps = 0
    total_loss = total_ae = total_svdd = total_center_loss = 0.0

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
            Num_batch = max(1, int(math.ceil(K * N)))

            model.zero_grad()
            outputs, code = model(x[:Num_batch, :])
            R = torch.mean(torch.sum((code - C_local) ** 2, dim=1))
            ae_loss = criterion(outputs, x[:Num_batch, :])
            train_loss = ae_loss + global_gamma * R
            train_loss.backward()
            with torch.no_grad():
                for p, cg, cl in zip(model.parameters(), c_global, c_local):
                    p.grad.add_(cg - cl)
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            with torch.no_grad():
                for p in model.parameters():
                    p.sub_(lr * p.grad)

            total_loss += train_loss.item()
            total_ae += ae_loss.item()
            total_svdd += R.item()
            last_R = R.item()
            batch_count += 1
            num_steps += 1

            if Num_batch < N:
                _, c_code = model(x[Num_batch:, :])
                if c_code.numel() > 0:
                    center = torch.mean(c_code, dim=0).detach()
                    update_center.zero_grad()
                    center_loss = criterion(C_local, center)
                    center_loss.backward()
                    update_center.step()
                    total_center_loss += center_loss.item()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    compute_time = time.perf_counter() - compute_start
    t_kp = compute_time / max(1, int(localEpochs))

    # Option II control-variate refresh + parameter/control deltas to send
    # back. Everything stays on `device` throughout the round.
    with torch.no_grad():
        delta_y, delta_c, new_c_local = [], [], []
        coef = 1.0 / max(1, num_steps * lr)
        for p, xg, cl, cg in zip(model.parameters(), x_global, c_local, c_global):
            dy = p.detach() - xg
            new_cl = cl - cg + coef * dy
            delta_y.append(dy)
            delta_c.append(new_cl - cl)
            new_c_local.append(new_cl)

    if batch_count == 0:
        avg_loss = avg_ae = avg_svdd = avg_center_loss = 0.0
    else:
        avg_loss = total_loss / batch_count
        avg_ae = total_ae / batch_count
        avg_svdd = total_svdd / batch_count
        avg_center_loss = total_center_loss / batch_count

    val_loss, val_ae, val_svdd, center_mse = dasvdd.DASVDD_validate(
        model, in_shape, C_local.detach(), val_loader, criterion, global_gamma, device
    )

    return delta_y, delta_c, new_c_local, client_n_samples, \
        {"train_loss": avg_loss, "ae_loss": avg_ae, "svdd_loss": avg_svdd,
         "compute_time": compute_time, "t_kp": t_kp}, \
        {"val_loss": val_loss, "val_ae_loss": val_ae, "val_svdd_loss": val_svdd, "val_center_mse": center_mse}, \
        avg_center_loss, C_local.detach().cpu(), last_R, global_gamma


def federated_train_scaffold(device, dataset, client_files, val_loader, test_loader, y_test,
                              in_shape, code_size, lr=1e-3, batch_size=256, global_rounds=100,
                              localEpochs=1, Adagard_lr=0.01, Adagard_lr_decay=0.0, weight_decay=1e-2,
                              K=0.8, scaffold_lr=None, grad_clip_norm=5.0, client_comm_times=None,
                              comm_bandwidth_mbps=100.0, seed=12, T=10, verbose=True):
    utils.set_random_seeds(seed)
    device = torch.device(device)
    local_lr = scaffold_lr if scaffold_lr is not None else lr
    nb_clients = len(client_files)
    client_loaders = []
    client_counts = []
    total_time = 0.0

    model_class = _get_model_class(dataset)

    for path in client_files:
        loader, n = dasvdd.load_client_loader(path, batch_size=batch_size, in_shape=in_shape,
                                               shuffle=True, device=device)
        client_loaders.append(loader)
        client_counts.append(n)

    if verbose:
        print(f"[SCAFFOLD] Computing local gamma for each client (T={T}):")
    client_gammas = []
    for i, loader in enumerate(client_loaders):
        g = dasvdd.tune_gamma(model_class, in_shape, code_size, loader, device, T=T)
        client_gammas.append(g)
        if verbose:
            print(f" Client {i}: gamma={g:.6f}")

    total_samples = sum(client_counts)
    if total_samples == 0:
        raise RuntimeError("No samples across clients.")
    weighted_gamma = sum(g * n for g, n in zip(client_gammas, client_counts)) / float(total_samples)
    if verbose:
        print("Initial aggregated global gamma:", weighted_gamma)

    global_model = model_class(input_shape=in_shape, code_size=code_size).to(device)
    global_state = global_model.state_dict()
    global_C = torch.randn(code_size, device=device, requires_grad=False)
    criterion = nn.MSELoss()

    # Control variates live on `device` throughout (DASVDD_test/eval doesn't
    # move the model, unlike autoencoder.get_reconstruction_error, but we
    # keep everything explicit here regardless for consistency).
    c_global = [torch.zeros_like(p) for p in global_model.parameters()]
    c_locals = [[torch.zeros_like(p) for p in global_model.parameters()] for _ in range(nb_clients)]

    if client_comm_times is None:
        per_client_comm = dasvdd.estimate_comm_time(global_model, comm_bandwidth_mbps, num_transfers=2)
        client_comm_times = [per_client_comm] * nb_clients
    else:
        client_comm_times = [float(c) for c in client_comm_times]

    model_state_bytes = dasvdd.state_dict_bytes(global_state)
    center_bytes = global_C.numel() * global_C.element_size()
    total_bytes = total_bytes_down = total_bytes_up = 0

    history = []
    for rnd in range(1, int(global_rounds) + 1):
        if verbose:
            print(f"\n[SCAFFOLD] --- Round {rnd}/{global_rounds} ---")

        global_state = global_model.state_dict()
        delta_y_sum = [torch.zeros_like(p) for p in global_model.parameters()]
        delta_c_sum = [torch.zeros_like(p) for p in global_model.parameters()]

        client_Cs, client_Rs = [], []
        client_train_metrics_list, client_val_metrics_list, client_center_loss_list = [], [], []
        client_compute_times = []

        for cid in range(nb_clients):
            local_model = model_class(input_shape=in_shape, code_size=code_size)
            local_model.load_state_dict(copy.deepcopy(global_state))
            local_model.to(device)

            local_C = global_C.clone().detach().to(device)
            local_C.requires_grad = True
            local_update_center = torch.optim.Adagrad([local_C], lr=Adagard_lr, lr_decay=Adagard_lr_decay)

            delta_y, delta_c, new_c_local, n, train_metrics, val_metrics, avg_center_loss, C_client, R_client, _ = \
                dasvdd_local_update_scaffold(
                    model=local_model, in_shape=in_shape, client_loader=client_loaders[cid],
                    val_loader=val_loader, client_n_samples=client_counts[cid], device=device,
                    lr=local_lr, update_center=local_update_center, K=K, localEpochs=localEpochs,
                    C_local=local_C, global_gamma=weighted_gamma, criterion=criterion,
                    c_global=c_global, c_local=c_locals[cid], grad_clip_norm=grad_clip_norm
                )

            w = client_counts[cid] / total_samples
            for i in range(len(delta_y_sum)):
                delta_y_sum[i] += delta_y[i] * w
                delta_c_sum[i] += delta_c[i] * w
            c_locals[cid] = new_c_local

            client_Cs.append(C_client)
            client_Rs.append(R_client)
            client_train_metrics_list.append(train_metrics)
            client_val_metrics_list.append(val_metrics)
            client_center_loss_list.append(avg_center_loss)
            client_compute_times.append(train_metrics["compute_time"])

            if verbose:
                print(f" client {cid}: n={n}, train_loss={train_metrics['train_loss']:.4f}, "
                      f"R={R_client:.6f}, C_norm={C_client.norm().item():.6f}")

            del local_model, local_update_center
            torch.cuda.empty_cache()

        client_t_k = [ct + cm for ct, cm in zip(client_compute_times, client_comm_times)]
        straggler = int(np.argmax(client_t_k))
        round_time = float(client_t_k[straggler])
        total_time += round_time

        round_bytes = 2 * nb_clients * (model_state_bytes + center_bytes)
        total_bytes_down += nb_clients * (model_state_bytes + center_bytes)
        total_bytes_up += nb_clients * (model_state_bytes + center_bytes)
        total_bytes += round_bytes

        new_global_state = collections.OrderedDict()
        for (k, v), dy in zip(global_state.items(), delta_y_sum):
            new_global_state[k] = v + dy.to(v.device)
        global_model.load_state_dict(new_global_state)

        c_global = [cg + dc for cg, dc in zip(c_global, delta_c_sum)]

        total = sum(client_counts)
        weighted_C = torch.zeros_like(client_Cs[0])
        for c, w in zip(client_Cs, client_counts):
            weighted_C += c * (w / total)
        global_C = weighted_C.to(device)
        global_C.requires_grad = False

        weighted_train_loss = sum(m['train_loss'] * n for m, n in zip(client_train_metrics_list, client_counts)) / total
        weighted_center_loss = sum(cl * n for cl, n in zip(client_center_loss_list, client_counts)) / total
        weighted_val_loss = sum(m['val_loss'] * n for m, n in zip(client_val_metrics_list, client_counts)) / total
        weighted_R = sum(r * n for r, n in zip(client_Rs, client_counts)) / total

        try:
            auc, ap, _ = dasvdd.DASVDD_test(global_model, global_C, in_shape, weighted_gamma,
                                             test_loader, y_test, criterion, device)
        except Exception as e:
            auc, ap = float('nan'), float('nan')
            if verbose:
                print("Warning: error in DASVDD_test:", e)

        wandb.log({
            "round": rnd,
            "weighted_train_loss": weighted_train_loss,
            "weighted_center_loss": weighted_center_loss,
            "weighted_val_loss": weighted_val_loss,
            "global_R": weighted_R,
            "global_gamma": weighted_gamma,
            "AUC_test": auc,
            "AP_test": ap,
            "scaffold_lr": local_lr,
            "round_time_sec": round_time,
            "total_time_sec": total_time,
            "straggler_client": straggler,
            "round_bytes": round_bytes,
            "total_bytes": total_bytes,
            "total_bytes_down": total_bytes_down,
            "total_bytes_up": total_bytes_up,
        })

        if verbose:
            print(f"[SCAFFOLD] Round {rnd} — AUC: {auc:.2f}, weighted_val_loss: {weighted_val_loss:.6f}, "
                  f"total_time={total_time:.4f}s")

        history.append({"round": rnd, "auc": auc, "ap": ap, "weighted_val_loss": weighted_val_loss,
                         "round_time_sec": round_time, "total_time_sec": total_time})

    return global_model, global_C, history, weighted_gamma
