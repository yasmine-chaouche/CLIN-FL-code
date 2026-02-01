import numpy as np
import pandas as pd
import random
import os           
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import clone_model
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import utils
import wandb

def build_autoencoder(input_dim):
    utils.set_random_seeds()
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

def load_client_data(filename):
    # set_random_seeds()
    df_train = pd.read_csv(filename)
    return df_train
def train_local_model(model, X_train, X_test):
    utils.set_random_seeds()
    model.compile(optimizer='Nadam', loss=MeanSquaredError())
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, X_train,
              epochs=5,
              batch_size=1024,
              validation_data=(X_test, X_test),
              shuffle=True,
              callbacks=[early_stop]
)
    return model.get_weights()

def train_centralized_model(model, X_train, X_test):
    utils.set_random_seeds()
    model.compile(optimizer='Nadam',loss=MeanSquaredError())
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, X_train,
                       epochs=360,
                       batch_size=1024,
                       validation_data=(X_test, X_test),
                       shuffle=True,
                       callbacks=[early_stop])
    return history

# Calculate reconstruction error
def get_reconstruction_error(model, X_test, y_test):
    reconstructions = model.predict(X_test, verbose=0)
    reconstruction_error = np.mean(np.square(X_test.values - reconstructions), axis=1)

    return pd.DataFrame({
        'error': reconstruction_error,
        'y_true': y_test
    }).reset_index(drop=True)

def federated_averaging(weights_list, sample_counts):
    utils.set_random_seeds()
    total_samples = sum(sample_counts)
    avg_weights = []
    for weights in zip(*weights_list):
        weighted_sum = sum(w * (count / total_samples) for w, count in zip(weights, sample_counts))
        avg_weights.append(weighted_sum)
    return avg_weights


def federated_learning(nb_clients, nb_rounds,split_type,X_test,y_test):
    utils.set_random_seeds()
    client_files = [f'{split_type}_split/AIF_{i}_{split_type}.csv' for i in range(nb_clients)]

    # Initialize global model
    sample_input = load_client_data(client_files[0])
    global_model = build_autoencoder(sample_input.shape[1])
    
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
            local_model.compile(optimizer='Nadam', loss=MeanSquaredError())
            history = local_model.fit(client_data, client_data,
                                      epochs=5,
                                      batch_size=1024,
                                      validation_data=(X_test, X_test),
                                      shuffle=True,
                                      callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                                      verbose=0)
            weights = local_model.get_weights()
            local_weights.append(weights)
            local_train_losses.append(history.history['loss'][-1])
            local_val_losses.append(history.history['val_loss'][-1])
        
        # Weighted aggregation of weights
        new_weights = federated_averaging(local_weights, sample_counts)
        global_model.set_weights(new_weights)
        
        # Calculate weighted train and validation losses
        total_samples = sum(sample_counts)
        weighted_train_loss = sum(loss * (count / total_samples) for loss, count in zip(local_train_losses, sample_counts))
        weighted_val_loss = sum(loss * (count / total_samples) for loss, count in zip(local_val_losses, sample_counts))
        
        recons_df = get_reconstruction_error(global_model, X_test, y_test)
        #Thresholf tuning
        thresh_df = utils.thresholdTuning(recons_df, 10)
        TEST_THRESHOLD = thresh_df[thresh_df['f1_score'] == thresh_df['f1_score'].max()]['threshold'].values[0]
        print("Results: =====================")
        print(f"Threshold: {TEST_THRESHOLD:.6f}")
        recons_df['y_pred'] = recons_df['error'] > TEST_THRESHOLD
        fpr, tpr, thresholds = roc_curve(recons_df['y_true'], recons_df['error'])
        roc_auc = auc(fpr, tpr)
        print("AUC_ROC: ",roc_auc)

        # Direct metric calculation
        acc = accuracy_score(recons_df['y_true'], recons_df['y_pred'])
        prec = precision_score(recons_df['y_true'], recons_df['y_pred'])
        rc = recall_score(recons_df['y_true'], recons_df['y_pred'])
        f1 = f1_score(recons_df['y_true'], recons_df['y_pred'])

        print("========== Final Metrics ==========")
        print(f"Accuracy: {acc*100:.3f}%")
        print(f"Precision: {prec*100:.3f}%")
        print(f"Recall: {rc*100:.3f}%")
        print(f"F1-Score: {f1*100:.3f}%")

        wandb.log({
            "round": rnd,
            "weighted_train_loss": weighted_train_loss,
            "weighted_val_loss": weighted_val_loss,
            "AUC_test": roc_auc,
            "F1score": f1
        })
    return global_model