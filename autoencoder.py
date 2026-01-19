

def federated_learning(nb_clients, nb_rounds,split_type,X_test,y_test):
    set_random_seeds()
    client_files = [f'{split_type}_split/AIF_{i}_{split_type}.csv' for i in range(nb_clients)]

    # Initialize global model
    sample_input = load_client_data(client_files[0])
    global_model = build_autoencoder(sample_input.shape[1])
    
    for rnd in range(nb_rounds):
        print(f"Round {rnd+1}/{nb_rounds}")
        local_weights = []
        sample_counts = []
        
        for client_file in client_files:
            client_data = load_client_data(client_file)
            num_samples = len(client_data)
            sample_counts.append(num_samples)
            local_model = clone_model(global_model)
            local_model.set_weights(global_model.get_weights())
            weights = train_local_model(local_model, client_data,X_test)
            local_weights.append(weights)
        
        # Weighted aggregation of weights
        new_weights = federated_averaging(local_weights, sample_counts)
        global_model.set_weights(new_weights)
        recons_df = get_reconstruction_error(global_model, X_test, y_test)
        # #Thresholf tuning
        # thresh_df = thresholdTuning(recons_df, 10)
        # TEST_THRESHOLD = thresh_df[thresh_df['f1_score'] == thresh_df['f1_score'].max()]['threshold'].values[0]
        # print("Results: =====================")
        # print(f"Threshold: {TEST_THRESHOLD:.6f}")
        # recons_df['y_pred'] = recons_df['error'] > TEST_THRESHOLD
        FUfpr,FUtpr,FUauc_roc=plot_auc_roc(recons_df)

        # # Direct metric calculation
        # acc = accuracy_score(recons_df['y_true'], recons_df['y_pred'])
        # prec = precision_score(recons_df['y_true'], recons_df['y_pred'])
        # rc = recall_score(recons_df['y_true'], recons_df['y_pred'])
        # f1 = f1_score(recons_df['y_true'], recons_df['y_pred'])

        # print("========== Final Metrics ==========")
        # print(f"Accuracy: {acc*100:.3f}%")
        # print(f"Precision: {prec*100:.3f}%")
        # print(f"Recall: {rc*100:.3f}%")
        # print(f"F1-Score: {f1*100:.3f}%")

    
    global_model.save(f"{split_type}_federated_global_model.h5")
    return global_model