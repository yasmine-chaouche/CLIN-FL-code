import pandas as pd
import cudf as cdf
import numpy as np
from numba import cuda
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import random
from scipy.stats import dirichlet
from sklearn.cluster import KMeans
from torch.backends import cudnn
import torch
import tensorflow as tf
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.decomposition import IncrementalPCA
import warnings
from cuml.decomposition import PCA  # GPU version
import cupy as cp
import shutil
from collections import defaultdict
from scipy.interpolate import interp1d
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Activation,Dropout
from keras import regularizers, initializers
from tensorflow.keras.losses import mae
from sklearn.metrics import (
    accuracy_score, recall_score, classification_report, 
    confusion_matrix, roc_curve, auc, f1_score, precision_score, matthews_corrcoef
)
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam,Nadam
from tensorflow.keras.models import clone_model
from tensorflow.keras import layers
# Ignore SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
# Ignore FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy.stats import t
import cupy as cp
import cudf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import subprocess
from sklearn.preprocessing import MinMaxScaler
import cudf
from tensorflow.keras.callbacks import ReduceLROnPlateau
import torch.nn.functional as F

def set_random_seeds(seed=2):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.default_rng(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_jsd_matrix(df):
    df_copy=df.copy()
    df_gpu = cudf.DataFrame.from_pandas(df_copy)
    feature_columns = df_copy.columns[:-1]  # Exclude 'AIFID'
    NbClients = df_copy['AIFID'].nunique()

    # Use CuPy to create a GPU-based JSD matrix
    jsd_matrix_gpu = cp.zeros((len(feature_columns), NbClients+1))
    # Compute global histograms once on GPU
    global_histograms = {}
    bins_dict = {}

    for feature in feature_columns:
        global_counts, bins = np.histogram(df_copy[feature], bins=20, density=True)
        global_proba = global_counts / global_counts.sum()
        global_histograms[feature] = cp.array(global_proba)  # Move to GPU
        bins_dict[feature] = bins  # Store bin edges

    # Compute JSD for each feature and client in parallel
    for i, feature in enumerate(feature_columns):
        global_proba = global_histograms[feature]
        bins = bins_dict[feature]
        
        for j, client in enumerate(range(NbClients)):
            client_data = df_gpu[df_gpu['AIFID'] == client][feature].to_pandas().dropna()
            client_counts, _ = np.histogram(client_data, bins=bins, density=True)
            client_proba = cp.array(client_counts / client_counts.sum())  # Move to GPU
            jsd_matrix_gpu[i, j] = jensenshannon(global_proba.get(), client_proba.get())  # Use GPU arrays
        jsd_matrix_gpu[i, j+1] = cp.mean(jsd_matrix_gpu[i, :j+1])  # Compute mean on GPU

    jsd_matrix = jsd_matrix_gpu.get()
    jsd_df = pd.DataFrame(jsd_matrix, columns=[f"AIF {c+1}" for c in range(NbClients)] + ["Avg"])
    return jsd_df

def deactivate_per_broker2(latency_broker_to_aif, broker_aif_map, deactivation_ratio):
    updated_latency_broker_to_aif = latency_broker_to_aif.copy()
    updated_broker_aif_map = {broker: aifs.copy() for broker, aifs in broker_aif_map.items()}

    # Step 1: Deactivate top-latency links per broker
    for broker, aif_list in broker_aif_map.items():
        latencies = [(aif, updated_latency_broker_to_aif[(broker, aif)]) for aif in aif_list]
        latencies.sort(key=lambda x: x[1], reverse=True)  # sort by latency descending
        num_to_deactivate = int(len(latencies) * deactivation_ratio)

        for aif, _ in latencies[:num_to_deactivate]:
            updated_latency_broker_to_aif.pop((broker, aif), None)
            updated_broker_aif_map[broker].remove(aif)

    # Step 2: Ensure each AIF is connected to at least one broker
    aif_connections = defaultdict(list)
    for (broker, aif) in updated_latency_broker_to_aif:
        aif_connections[aif].append(broker)

    all_aifs = {aif for aifs in broker_aif_map.values() for aif in aifs}
    disconnected_aifs = {aif for aif in all_aifs if len(aif_connections[aif]) == 0}

    for aif in disconnected_aifs:
        # Find the broker with the lowest latency to this AIF
        candidate_links = [(broker, latency_broker_to_aif[(broker, aif)])
                           for broker in broker_aif_map
                           if (broker, aif) in latency_broker_to_aif]

        if candidate_links:
            best_broker, _ = min(candidate_links, key=lambda x: x[1])
            updated_latency_broker_to_aif[(best_broker, aif)] = latency_broker_to_aif[(best_broker, aif)]
            updated_broker_aif_map[best_broker].append(aif)

    print(f"%%%% Total Deactivated links {len(latency_broker_to_aif)-len(updated_latency_broker_to_aif)} out of {len(latency_broker_to_aif)} %%%%")

    return updated_broker_aif_map, updated_latency_broker_to_aif

def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return mean, h

def Conv_split(traindata, n_clients,seed,save_dir='./Conv_split'):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir) 
    os.makedirs(save_dir)  
    traindata_with_labels = traindata.copy()
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clients, init='k-means++',random_state=seed)
    cluster_labels = kmeans.fit_predict(traindata_with_labels)
    # Temporarily add cluster labels to the data
    traindata_with_labels['AIFID'] = cluster_labels
    # Save one CSV per client
    for client_id in range(n_clients):
        client_data = traindata_with_labels[traindata_with_labels['AIFID'] == client_id].drop('AIFID', axis=1)
        client_data.to_csv(f'./Conv_split/AIF_{client_id}_Conv.csv', index=False)

def set_pin_memory_for_device(device):
    return (device.type == 'cuda')

def thresholdTuning(df, iterations):
    thresh_df = {
        'threshold': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'mcc': [],
    }
    for i in range(1, iterations):  # Avoid 0% quantile (min value)
        thresh_value = df['error'].quantile(i / iterations)
        preds = df['error'] > thresh_value

        # Use binary metrics directly
        acc = accuracy_score(df['y_true'], preds)
        prec = precision_score(df['y_true'], preds)
        rc = recall_score(df['y_true'], preds)
        f1 = f1_score(df['y_true'], preds)
        mcc = matthews_corrcoef(df['y_true'], preds) 

        thresh_df['threshold'].append(thresh_value)
        thresh_df['accuracy'].append(acc)
        thresh_df['precision'].append(prec)
        thresh_df['recall'].append(rc)
        thresh_df['f1_score'].append(f1)
        thresh_df['mcc'].append(mcc)
        #print(f"Threshold: {thresh_value}\tAccuracy: {acc}\tPrecision: {prec}\tRecall: {rc}\tF1 Score: {f1}\tMCC: {mcc}")

    return pd.DataFrame(thresh_df)