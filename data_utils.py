import warnings
import numpy as np
import pandas as pd
import cudf
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
from cuml.decomposition import PCA

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


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

def apply_pca_with_cuml(df, n_components, pca_model=None):
    # if isinstance(df, pd.DataFrame):
    #     df = cudf.DataFrame.from_pandas(df)
    
    if pca_model is None:
        # Fit PCA on training data
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(df)
        
        # Get variance retained
        variance_ratios = pca.explained_variance_ratio_
        variance_ratios_np = variance_ratios.to_numpy()
        total_variance_retained = np.sum(variance_ratios_np)
        print(f"Retained {total_variance_retained * 100:.2f}% of the original variance "
              f"across {n_components} components")
        
        return (reduced.to_pandas() if hasattr(reduced, "to_pandas") else reduced), pca
    else:
        # Apply fitted PCA to test data
        reduced = pca_model.transform(df)
        return (reduced.to_pandas() if hasattr(reduced, "to_pandas") else reduced)

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
    df_test_scaled = scaler.transform(df_x_test)

    # Convert back to DataFrame and keep original column names
    df_train = pd.DataFrame(df_train_scaled, columns=df_train.columns)
    df_test = pd.DataFrame(df_test_scaled, columns=df_x_test.columns)

    return df_train, df_test,y_test

def data_preprocessing_CICIDS2017(data, scaler=None, fit_scaler=True):
    data_copy = data.copy()
    label_col = data_copy.columns[-1]
    
    # Convert labels: BENIGN=0, anything else (attacks)=1
    y = (data_copy[label_col] != 'BENIGN').astype(int)
    
    # Remove label column from features
    X = data_copy.drop(columns=[label_col])
    
    # Remove constant columns
    num_unique = X.nunique()
    one_variable_cols = num_unique[num_unique == 1].index
    X.drop(columns=one_variable_cols, inplace=True)
    
    # Handle infinity and NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(numeric_only=True), inplace=True)
    
    if fit_scaler:
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        return X_scaled, y, scaler
    else:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
        return X_scaled, y