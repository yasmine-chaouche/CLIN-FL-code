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

def plot_jsd_matrix(jsd_df):
    # Plot the JSD matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(jsd_df, cmap="Reds", annot=True, fmt=".3f")
    plt.xlabel("Clients")
    plt.ylabel("Features")
    plt.title("Jensen-Shannon Divergence Matrix: The divergence of client i from the global state")
    plt.show()

def load_and_preprocess_5G3E_data():
    # Load datasets and convert to pandas
    df_train = cudf.read_csv('./datasets/5G3Ev1/dataset_end_total_preprocessed.csv').to_pandas()
    df_test = cudf.read_csv('./datasets/5G3Ev1/dataset_test_full_server_1_packect_loss_preprocessed.csv').to_pandas()
    print("df_train.shape: ",df_train.shape)
    print("df_test.shape: ", df_test.shape)
    # df_train=df_train.iloc[:len(df_train)//3] # First 1/3
    # df_train=df_train.iloc[len(df_train)//3 : 2*len(df_train)//3]   # Second 1/3
    # df_train=df_train.iloc[2*len(df_train)//3 :]  #Last 1/3
    
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



def apply_feature_selection_with_variance_threshold(data):
    new_data = data.loc[:, VarianceThreshold(threshold=0.09).fit(data).get_support()]
    return new_data

def build_autoencoder_5G3E(input_dim):
    set_random_seeds()
    input_layer = keras.Input(shape=(input_dim,))

    # Encoder
    encoded = layers.Dense(64, activation="relu")(input_layer)
    encoded = layers.Dense(16, activation="relu")(encoded)  # Bottleneck

    # Decoder
    decoded = layers.Dense(64, activation="relu")(encoded)

    output_layer = layers.Dense(input_dim, activation="linear")(decoded)

    autoencoder = keras.Model(inputs=input_layer, outputs=output_layer)
    return autoencoder

def build_autoencoder_CICIDS2017(input_dim):
    set_random_seeds()
    input_layer = Input(shape=(input_dim,))
    layer = Dense(64, activation='relu', kernel_initializer='he_normal')(input_layer)
    layer = Dense(32, activation='relu', kernel_initializer='he_normal')(layer)
    layer = Dense(16, activation='relu', kernel_initializer='he_normal')(layer)
    layer = Dense(8, activation='relu', kernel_initializer='he_normal')(layer)
    layer = Dense(16, activation='relu', kernel_initializer='he_normal')(layer)
    layer = Dense(32, activation='relu', kernel_initializer='he_normal')(layer)
    layer = Dense(64, activation='relu', kernel_initializer='he_normal')(layer)
    output_layer = Dense(input_dim, activation='linear')(layer)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    return autoencoder

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.initializers import HeNormal
import tensorflow.keras.backend as K

# === Memory Module Layer ===
class MemoryModule(Layer):
    def __init__(self, mem_dim=50, mem_feat_dim=8, **kwargs):
        super(MemoryModule, self).__init__(**kwargs)
        self.mem_dim = mem_dim
        self.mem_feat_dim = mem_feat_dim

    def build(self, input_shape):
        # Memory items: shape (mem_dim, mem_feat_dim)
        self.memory = self.add_weight(
            shape=(self.mem_dim, self.mem_feat_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='memory_bank'
        )

    def call(self, inputs):
        # inputs: shape (batch_size, mem_feat_dim)
        # Attention weights: softmax over dot product with memory items
        similarity = tf.matmul(inputs, self.memory, transpose_b=True)  # (batch_size, mem_dim)
        attention = tf.nn.softmax(similarity, axis=1)                  # (batch_size, mem_dim)
        memory_output = tf.matmul(attention, self.memory)             # (batch_size, mem_feat_dim)
        return memory_output  # reconstructed features

# === MemAE Model for CICIDS2017 ===
def build_memAE_CICIDS2017(input_dim, mem_dim=50):
    set_random_seeds()
    He = HeNormal()
    input_layer = Input(shape=(input_dim,))

    # Encoder
    x = Dense(64, activation='relu', kernel_initializer=He)(input_layer)
    x = Dense(32, activation='relu', kernel_initializer=He)(x)
    x = Dense(16, activation='relu', kernel_initializer=He)(x)
    latent = Dense(8, activation='relu', kernel_initializer=He)(x)

    # Memory Module
    memory_output = MemoryModule(mem_dim=mem_dim, mem_feat_dim=8)(latent)

    # Decoder
    x = Dense(16, activation='relu', kernel_initializer=He)(memory_output)
    x = Dense(32, activation='relu', kernel_initializer=He)(x)
    x = Dense(64, activation='relu', kernel_initializer=He)(x)
    output_layer = Dense(input_dim, activation='linear')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

#=========== IAEAD Model for CICIDS2017 ==============
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
import tensorflow.keras.backend as K
from tensorflow.keras.layers import BatchNormalization

def build_IAEAD_CICIDS2017(input_dim, latent_dim=8):
    set_random_seeds()
    inputs = Input(shape=(input_dim,))

    # ----- Encoder -----
    x = Dense(64, activation='relu', kernel_initializer='he_normal')(inputs)
    x = Dense(32, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
    z = Dense(latent_dim, activation=None, name='latent')(x)  # latent space (for SVDD)

    # ----- Decoder -----
    x = Dense(16, activation='relu', kernel_initializer='he_normal')(z)
    x = Dense(32, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    outputs = Dense(input_dim, activation='linear')(x)


    # Full autoencoder model and encoder model
    return Model(inputs, outputs, name="IAEAD_Autoencoder"), Model(inputs, z, name="IAEAD_Encoder")


class IAEADLoss(tf.keras.losses.Loss):
    def __init__(self, model, encoder, c, R, lambda_svdd=1e-4, nu=0.1):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.c = c
        self.R = R
        self.lambda_svdd = lambda_svdd
        self.nu = nu

    def call(self, y_true, y_pred):
        recon_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
        z = self.encoder(y_true)
        dist = tf.reduce_sum(tf.square(z - self.c), axis=1)
        svdd_loss = self.lambda_svdd * (self.R**2 + (1.0 / self.nu) * tf.reduce_mean(tf.maximum(dist - self.R**2, 0.0)))
        return recon_loss + svdd_loss


def initialize_center_c(encoder, X_train, eps=0.1):
    latent = encoder.predict(X_train, batch_size=1024)
    c = np.mean(latent, axis=0)
    # c[np.abs(c) < eps] = eps * np.sign(c[np.abs(c) < eps])
    # return tf.constant(c, dtype=tf.float32)
    c[(np.abs(c) < eps) & (c < 0)] = -eps
    c[(np.abs(c) < eps) & (c > 0)] = eps
    return c

def update_radius(distances, nu):
    return np.quantile(np.sqrt(distances), 1 - nu)

#=================================================================================================
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import wandb
from tensorflow.keras.callbacks import Callback


class WandbAUCLoggingCallback(Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        # Use your custom reconstruction error function
        recons_df = get_reconstruction_error(self.model, self.X_val, self.y_val)

        try:
            fpr, tpr, thresholds = roc_curve(recons_df['y_true'], recons_df['error'])
            roc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(recons_df['y_true'], recons_df['error'])
            pr_auc = auc(recall, precision)
        except Exception as e:
            print(f"[WARN] AUC computation failed: {e}")
            roc, pr_auc = 0.0, 0.0

        wandb.log({
            "epoch": epoch,
            "train_loss": logs.get("loss"),
            "val_loss": logs.get("val_loss"),
            "roc_auc": roc,
            "pr_auc": pr_auc
        })

from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
import wandb

class WandbIAEADAUCLoggingCallback(Callback):
    def __init__(self, X_val, y_val, encoder, centroid_tensor):
        """
        encoder: Keras model that outputs latent representations
        centroid_tensor: tf.Tensor or np.array representing the center c
        """
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.encoder = encoder
        self.centroid = centroid_tensor if isinstance(centroid_tensor, np.ndarray) else centroid_tensor.numpy()

    def compute_scores(self):
        recon = self.model.predict(self.X_val, verbose=0)
        latent = self.encoder.predict(self.X_val, verbose=0)

        # Reconstruction score (L1)
        l1_loss = np.mean(np.abs(recon - self.X_val), axis=1)
        recon_scores = l1_loss#1 - ((l1_loss - l1_loss.min()) / (1e-8 + (l1_loss.max() - l1_loss.min())))

        # DEC-style latent score (distance-based)
        centroid = np.mean(latent, axis=0, keepdims=True)
        # compute squared distance to centroid
        dist = np.sum((latent - self.centroid) ** 2, axis=1)

        # normalize distance
        dist_norm = dist / (dist.max() + 1e-8)

        # q distribution
        q_pos = 1.0 / (1.0 + 5 * dist_norm)
        q_neg = 1 - q_pos
        q = np.stack([q_pos, q_neg], axis=1)

        # normalize q like in torch code
        q = (q.T / np.sum(q, axis=1)).T

        # compute weights
        weight = q ** 2 / np.sum(q, axis=0)

        # normalize weights to get p
        p = (weight.T / np.sum(weight, axis=1)).T

        # DEC anomaly score
        dec_scores = p[:, 0]

        return recon_scores, dec_scores

    def on_epoch_end(self, epoch, logs=None):
        try:
            recon_scores, dec_scores = self.compute_scores()

            # --- Reconstruction AUC ---
            roc_rec = roc_auc_score(self.y_val, recon_scores)
            prec_rec, rec_rec, _ = precision_recall_curve(self.y_val, recon_scores)
            pr_auc_rec = auc(rec_rec, prec_rec)

            # --- DEC AUC ---
            roc_dec = roc_auc_score(self.y_val, dec_scores)
            prec_dec, rec_dec, _ = precision_recall_curve(self.y_val, dec_scores)
            pr_auc_dec = auc(rec_dec, prec_dec)

        except Exception as e:
            print(f"[WARN] AUC computation failed: {e}")
            roc_rec, pr_auc_rec, roc_dec, pr_auc_dec = 0.0, 0.0, 0.0, 0.0

        wandb.log({
            "epoch": epoch,
            "train_loss": logs.get("loss"),
            "val_loss": logs.get("val_loss"),
            "roc_auc_rec": roc_rec,
            "pr_auc_rec": pr_auc_rec,
            "roc_auc_dec": roc_dec,
            "pr_auc_dec": pr_auc_dec,
        })


def train_centralized_model(model, X_train, X_test, y_test, epochs, batch_size, learning_rate, optimizer_name, loss_fn):
    set_random_seeds()
    if optimizer_name == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'Nadam':
        optimizer = 'Nadam'
    elif optimizer_name == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    encoder = keras.Model(inputs=model.input, outputs=model.get_layer("latent").output)  # depends on your layer name
    centroid = tf.reduce_mean(encoder(X_train), axis=0, keepdims=True)
    model.compile(optimizer=optimizer, loss=loss_fn)
    early_stop = EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(
                                monitor='val_loss',   # metric to watch
                                patience=5,               # epochs with no improvement before reducing LR
                                min_lr=1e-7,              # minimum LR allowed
                                verbose=1,                # show messages when LR is reduced
                                factor=0.1                # reduce LR by a factor of 10
                            )
    auc_callback = WandbIAEADAUCLoggingCallback(X_val=X_test, y_val=y_test, encoder=encoder, centroid_tensor=centroid)

    history = model.fit(X_train, X_train,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_data=(X_test, X_test),
                       shuffle=True,
                       callbacks=[early_stop, auc_callback]
                       )
    return history

def get_reconstruction_error(model, X_test, y_test):
    reconstructions = model.predict(X_test, verbose=0)
    reconstruction_error = np.mean(np.square(X_test.values - reconstructions), axis=1)

    return pd.DataFrame({
        'error': reconstruction_error,
        'y_true': y_test.values
    }).reset_index(drop=True)

def plot_confusion_matrix(recons_df):
    cm = confusion_matrix(recons_df['y_true'],recons_df['y_pred'])
    labels=['normal','anomaly']
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, cmap='Blues', cbar=False, fmt='.6g',xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

def plot_auc_roc(recons_df):
    fpr, tpr, thresholds = roc_curve(recons_df['y_true'], recons_df['error'])
    roc_auc = auc(fpr, tpr)
    print("AUC_ROC: ",roc_auc)
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    return fpr, tpr, roc_auc

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

def kmeans_split(traindata, n_clients,seed,save_dir='./kmeans_split'):
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
        client_data.to_csv(f'./kmeans_split/AIF_{client_id}_kmeans.csv', index=False)

def uniform_split(traindata, n_clients,save_dir='./uniform_split'):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir) 
    os.makedirs(save_dir) 

    traindata_with_labels = traindata.copy()
    traindata_with_labels['AIFID'] =np.random.randint(0, n_clients, size=traindata.shape[0])

    for client_id in range(n_clients):
        client_data = traindata_with_labels[traindata_with_labels['AIFID'] == client_id].drop(columns=['AIFID'])
        client_data.to_csv(f'./uniform_split/AIF_{client_id}_uniform.csv', index=False)

def load_client_data(filename):
    # set_random_seeds()
    df_train = pd.read_csv(filename)
    return df_train

def train_local_model(model, X_train, X_test, batch_size, epochs, optimizer_name, learning_rate, loss_fn, patience=10):
    set_random_seeds()
    if optimizer_name == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'Nadam':
        optimizer = 'Nadam'
    elif optimizer_name == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    model.compile(optimizer=optimizer, loss=loss_fn)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    
    history = model.fit(X_train, X_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test, X_test),
                        shuffle=True,
                        callbacks=[early_stop],
                        verbose=0)
    
    return model.get_weights(), history.history['loss'][-1], history.history['val_loss'][-1]



def federated_averaging(weights_list, sample_counts):
    set_random_seeds()
    total_samples = sum(sample_counts)
    avg_weights = []
    for weights in zip(*weights_list):
        weighted_sum = sum(w * (count / total_samples) for w, count in zip(weights, sample_counts))
        avg_weights.append(weighted_sum)
    return avg_weights

def federated_learning(nb_clients, nb_rounds, split_type, X_test, y_test, dataset, batch_size, epochs, optimizer_name, learning_rate, latent_dim=8, lambda_svdd=1e-4, nu=0.1):
    set_random_seeds()
    global_train_losses = []
    global_val_losses = []
    print("Starting Federated Learning...")
    client_files = [f'{split_type}_split/AIF_{i}_{split_type}.csv' for i in range(nb_clients)]

    # Initialize global model
    sample_input = load_client_data(client_files[0])
    if dataset == '5G3E':
        global_model = build_autoencoder_5G3E(sample_input.shape[1])
    elif dataset == 'CICIDS':
        global_model , encoder = build_IAEAD_CICIDS2017(sample_input.shape[1], latent_dim=latent_dim)

        # Initialize SVDD center using global encoder
    all_train_data = np.vstack([load_client_data(f) for f in client_files])
    c = initialize_center_c(encoder, all_train_data)
    R = tf.Variable(1.0, dtype=tf.float32)

    # Loss fn shared across clients
    loss_fn = IAEADLoss(global_model, encoder, c, R, lambda_svdd=lambda_svdd, nu=nu)
    
    for rnd in range(nb_rounds):
        print(f"Round {rnd+1}/{nb_rounds}")
        local_weights = []
        sample_counts = []
        local_train_losses = []
        local_val_losses = []
        
        for client_file in client_files:
            client_data = load_client_data(client_file)
            num_samples = len(client_data)
            sample_counts.append(num_samples)
            local_model = clone_model(global_model)
            local_model.set_weights(global_model.get_weights())
            weights, train_loss, val_loss = train_local_model(
                        local_model,
                        client_data,
                        X_test,
                        batch_size=batch_size,
                        epochs=epochs,
                        optimizer_name=optimizer_name,
                        learning_rate=learning_rate,
                        loss_fn=loss_fn
                    )
            local_weights.append(weights)
            local_train_losses.append(train_loss)
            local_val_losses.append(val_loss)
        
        # Weighted aggregation of weights
        new_weights = federated_averaging(local_weights, sample_counts)
        global_model.set_weights(new_weights)

        # Compute global train/val loss using weighted average
        total_samples = sum(sample_counts)
        weighted_train_loss = sum(l * n for l, n in zip(local_train_losses, sample_counts)) / total_samples
        weighted_val_loss = sum(l * n for l, n in zip(local_val_losses, sample_counts)) / total_samples
        global_train_losses.append(weighted_train_loss)
        global_val_losses.append(weighted_val_loss)

        # Evaluate global model with updated encoder
        encoder = keras.Model(inputs=global_model.input, outputs=global_model.get_layer("latent").output)
        recon = global_model.predict(X_test, verbose=0)
        latent = encoder.predict(X_test, verbose=0)

        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

        # l1_loss = np.mean(np.abs(recon - X_test), axis=1)
        recon_loss = np.mean(np.square(X_test - recon), axis=1)
        recon_scores = recon_loss  #1 - ((recon_loss - recon_loss.min()) / (1e-8 + (recon_loss.max() - recon_loss.min())))

        #Score de décision
        dist = np.sum((latent - c) ** 2, axis=1)

        # normalize distance
        dist_norm = dist / (dist.max() + 1e-8)

        # q distribution
        q_pos = 1.0 / (1.0 + 5 * dist_norm)
        q_neg = 1 - q_pos
        q = np.stack([q_pos, q_neg], axis=1)

        # normalize q like in torch code
        q = (q.T / np.sum(q, axis=1)).T

        # compute weights
        weight = q ** 2 / np.sum(q, axis=0)

        # normalize weights to get p
        p = (weight.T / np.sum(weight, axis=1)).T

        # DEC anomaly score
        dec_scores = p[:, 0]

        # AUC metrics
        try:    
            roc_rec = roc_auc_score(y_test, recon_scores)
            prec_rec, rec_rec, _ = precision_recall_curve(y_test, recon_scores)
            pr_auc_rec = auc(rec_rec, prec_rec)

            roc_dec = roc_auc_score(y_test, dec_scores)
            prec_dec, rec_dec, _ = precision_recall_curve(y_test, dec_scores)
            pr_auc_dec = auc(rec_dec, prec_dec)
        except Exception as e:
            print(f"[WARN] AUC failed: {e}")
            roc_rec, pr_auc_rec, roc_dec, pr_auc_dec = 0, 0, 0, 0

        # wandb log
        wandb.log({
            "round": rnd,
            "global_train_loss": weighted_train_loss,
            "global_val_loss": weighted_val_loss,
            "roc_auc_rec": roc_rec,
            "pr_auc_rec": pr_auc_rec,
            "roc_auc_dec": roc_dec,
            "pr_auc_dec": pr_auc_dec
        })

    global_model.save(f"{split_type}_federated_global_model_IAEAD.h5")
    return global_model

def deactivate_per_broker(latency_broker_to_aif, broker_aif_map, deactivation_ratio):
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

def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return mean, h


import math
def _bias_to_block(bias_vector):
    """
    Turn 1D bias vector into a near-square 2D block (pad with zeros if needed).
    Returns the 2D array and its original length.
    """
    size = bias_vector.size
    side = int(math.ceil(math.sqrt(size)))
    padded_len = side * side
    padded = np.zeros(padded_len, dtype=bias_vector.dtype)
    padded[:size] = bias_vector
    block = padded.reshape(side, side)
    return block, size

def plot_encoder_biases(model, include_decoder=False, cmap="seismic", figsize=(10,4), save_path=None):
    """
    Plots bias vectors for encoder layers of AE_MNIST as a concatenated heatmap.
    - include_decoder: if True, also include decoder biases after encoder.
    - cmap: matplotlib colormap
    - save_path: if provided, saves the figure to that path
    """
    # Collect encoder layers in the order you want to visualize
    bias_tensors = [
        model.encoder_hidden_layer.bias,   # b1
        model.encoder_middle.bias,         # b2
        model.encoder_output_layer.bias    # b3
    ]

    if include_decoder:
        bias_tensors += [
            model.decoder_hidden_layer.bias,  # b4
            model.decoder_middle.bias,        # b5
            model.decoder_output_layer.bias   # b6
        ]

    # Convert to numpy and build blocks
    blocks = []
    orig_lengths = []
    for t in bias_tensors:
        b_np = t.detach().cpu().numpy().ravel()
        block, length = _bias_to_block(b_np)
        blocks.append(block)
        orig_lengths.append(length)

    # Find max rows among blocks and pad each block vertically to match
    max_rows = max(block.shape[0] for block in blocks)
    padded_blocks = []
    widths = []
    for block in blocks:
        rows, cols = block.shape
        if rows < max_rows:
            pad_rows = max_rows - rows
            # pad bottom with zeros
            block_padded = np.vstack([block, np.zeros((pad_rows, cols), dtype=block.dtype)])
        else:
            block_padded = block
        padded_blocks.append(block_padded)
        widths.append(block_padded.shape[1])

    # Concatenate horizontally
    concatenated = np.concatenate(padded_blocks, axis=1)

    # Plot
    plt.figure(figsize=figsize)
    im = plt.imshow(concatenated, cmap=cmap, aspect="auto")
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel("Bias value", rotation=270, labelpad=12)

    # Draw vertical separators between blocks and tick labels centered under each block
    cum_width = 0
    xticks = []
    xlabels = []
    for i, w in enumerate(widths):
        # separator before next block (except after last)
        cum_width += w
        if i != len(widths) - 1:
            plt.axvline(x=cum_width - 0.5, color="k", linewidth=1)
        # center position for label of this block
        center = cum_width - w / 2.0
        xticks.append(center)
        # name labels b1..b6 (or up to b3)
        xlabels.append(f"b{i+1}")

    plt.xticks(xticks, xlabels)
    plt.yticks([])  # usually not helpful to show row numbers
    plt.title("Bias visualization (reshaped & concatenated)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

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