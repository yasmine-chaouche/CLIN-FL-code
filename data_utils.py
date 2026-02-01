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


def load_latency_data(filepath):
    df = cudf.read_csv(filepath).to_pandas()
    df = df[df["serv_provider"] == "Google/Youtube"].copy()
    df["delay_syn_synack"] = (df["delay_syn_synack"] / 1000).round(3)
    df["delay_synack_ack"] = (df["delay_synack_ack"] / 1000).round(3)
    df["back-hauling-delay"] = (df["delay_synack_ack"] - df["delay_syn_synack"]).round(3)
    return df[df["back-hauling-delay"].between(20, 40)]

def get_inverse_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.linspace(0, 1, len(sorted_data), endpoint=False)
    return interp1d(cdf, sorted_data, bounds_error=False, fill_value=(sorted_data[0], sorted_data[-1]))

def apply_pca_with_cuml(df, n_components):
    if isinstance(df, pd.DataFrame):
        df = cudf.DataFrame.from_pandas(df)
    # Fit PCA
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(df)
    # Get variance retained
    variance_ratios = pca.explained_variance_ratio_
    variance_ratios_np = variance_ratios.values_host  # cuDF Series → NumPy array
    total_variance_retained = np.sum(variance_ratios_np)
    print(f"Retained {total_variance_retained * 100:.2f}% of the original variance "
          f"across {n_components} components")

    return reduced.to_pandas()

def load_and_preprocess_5G3E_data():
    # Load datasets and convert to pandas
    df_train = cudf.read_csv('./datasets/5G3Ev1/dataset_end_total_preprocessed.csv').to_pandas()
    df_test = cudf.read_csv('./datasets/5G3Ev1/dataset_test_full_server_1_packect_loss_preprocessed.csv').to_pandas()
    print("df_train.shape: ",df_train.shape)
    print("df_test.shape: ", df_test.shape)
    
    print("Train and test datasets loaded...")
    # Drop unnecessary columns
    df_train = df_train.drop(columns=['Unnamed: 0'], errors='ignore')
    df_test = df_test.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'node_vmstat_pgfault_server_1'], errors='ignore')

    df_x_test = df_test.drop(columns='attack')
    y_test = df_test['attack'].to_numpy().astype(int)

    # Apply MinMaxScaler normalization
    scaler = MinMaxScaler()
    df_train_scaled = scaler.fit_transform(df_train)
    df_test_scaled = scaler.fit_transform(df_x_test)

    # Convert back to DataFrame and keep original column names
    df_train = pd.DataFrame(df_train_scaled, columns=df_train.columns)
    df_test = pd.DataFrame(df_test_scaled, columns=df_x_test.columns)

    return df_train, df_test,y_test

def data_preprocessing_CICIDS2017(data):
    data_copy = data.copy()

    # Label encoding: assuming last column is the label
    label_col = data_copy.columns[-1]
    data_copy[label_col] = data_copy[label_col].map({'BENIGN': 0}).fillna(1).astype(int)

    # Dropping columns with only one unique value
    num_unique = data_copy.nunique()
    one_variable_cols = num_unique[num_unique == 1].index
    data_copy.drop(columns=one_variable_cols, inplace=True)

    # Replacing infinite values with NaN
    data_copy.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Filling missing values with the median
    data_copy.fillna(data_copy.median(numeric_only=True), inplace=True)

    # Standardizing the dataset
    scaler = MinMaxScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data_copy), columns=data_copy.columns)

    return scaled_data