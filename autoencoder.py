import numpy as np
import pandas as pd
import random
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_curve, auc
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanSquaredError
import utils


def build_autoencoder(input_dim):
    utils.set_random_seeds()
    input_layer = keras.Input(shape=(input_dim,))

    # Encoder
    encoded = layers.Dense(64, activation="relu")(input_layer)
    encoded = layers.Dense(16, activation="relu")(encoded)  # Bottleneck

    # Decoder
    decoded = layers.Dense(64, activation="relu")(encoded)

    output_layer = layers.Dense(input_dim, activation="linear")(decoded)

    autoencoder = keras.Model(inputs=input_layer, outputs=output_layer)
    return autoencoder

def get_reconstruction_error(model, X_test, y_test):
    reconstructions = model.predict(X_test, verbose=0)
    reconstruction_error = np.mean(np.square(X_test.values - reconstructions), axis=1)

    return pd.DataFrame({
        'error': reconstruction_error,
        'y_true': y_test
    }).reset_index(drop=True)

def train_local_model(model, X_train, X_test):
    utils.set_random_seeds()
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001))
    early_stop = utils.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, X_train,
              epochs=5,
              batch_size=1024,
              validation_data=(X_test, X_test),
              shuffle=True,
              callbacks=[early_stop])
    return model.get_weights()

def federated_averaging(weights_list, sample_counts):
    utils.set_random_seeds()
    total_samples = sum(sample_counts)
    avg_weights = []
    for weights in zip(*weights_list):
        weighted_sum = sum(w * (count / total_samples) for w, count in zip(weights, sample_counts))
        avg_weights.append(weighted_sum)
    return avg_weights


def federated_learning(client_files, nb_rounds,X_test,y_test):
    utils.set_random_seeds()

    # Initialize global model
    sample_input = pd.read_csv(client_files[0])
    global_model = build_autoencoder(sample_input.shape[1])
    
    for rnd in range(nb_rounds):
        print(f"Round {rnd+1}/{nb_rounds}")
        local_weights = []
        sample_counts = []
        
        for client_file in client_files:
            client_data = pd.read_csv(client_file)
            num_samples = len(client_data)
            sample_counts.append(num_samples)
            local_model = clone_model(global_model)
            local_model.set_weights(global_model.get_weights())
            weights = train_local_model(local_model, client_data, X_test)
            local_weights.append(weights)
        
        # Weighted aggregation of weights
        new_weights = federated_averaging(local_weights, sample_counts)
        global_model.set_weights(new_weights)
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

    
    #global_model.save(f"{split_type}_federated_global_model.h5")
    return global_model