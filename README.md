# CLIN-FL: Cost-efficient Load-balancing for In-Network Federated Learning with non-IID data

[![Python Version](https://img.shields.io/badge/python-3.12.8-blue)](https://www.python.org/downloads/release/python-3128/)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.comnet.2026.112832-blue)](https://doi.org/10.1016/j.comnet.2026.112832)

This repository provides the official implementation of the framework presented in:  
**"CLIN-FL: Cost-efficient Load-balancing for In-Network Federated Learning with non-IID data"**  
*Yasmine Chaouche, Patient Ntumba, Stefano Secci, Pedro B. Velloso (Cnam, Paris, France)*  
*Computer Networks*, vol. 290, art. 112832, Elsevier, 2026. [https://doi.org/10.1016/j.comnet.2026.112832](https://doi.org/10.1016/j.comnet.2026.112832)

## 📄 Abstract

Federated Learning allows training machine learning models across many devices without sharing their raw data, which makes it attractive for deployment in telecommunication and in-network computing environments. However, real-world network data are often heterogeneous, and feature distribution discrepancies across clients can severely degrade the performance of the global model. While previous works have made notable progress by customizing the aggregation function and designing algorithmic training solutions, limited attention has been directed toward data-centric in-network load balancing techniques that cope with data feature distributions. In this paper, we propose a data load-balancing solution, named CLIN-FL, reducing feature distribution discrepancy thanks to in-network FL by controlling connectivity and latency. We evaluate our approach using both a standard Autoencoder model and a Deep Support Vector Data Descriptor (Deep SVDD) based model. Our results show that the proposed CLIN-FL schemes significantly improve feature distribution alignment while enabling a tunable trade-off between anomaly detection accuracy and communication cost, with improvements of up to 38.7% in AUC-ROC (Area Under the ROC Curve) and reductions of up to 20% in communication cost.

---

## 📊 Datasets
We evaluate our framework using two major network traffic datasets:

1.  **5G3Evl (5G End-to-End Emulation):** Time-series data from a 5G network testbed covering radio, computing, and network resources. It includes anomalies like CPU overload, link failures, and packet loss[cite: 585, 587].
2.  **CIC-IDS-2017:** A comprehensive dataset for evaluating intrusion detection systems, featuring real-world network traffic with attacks like DDoS, SQL Injection, and Brute Force.

### Network operator architecture emulation dataset:
This framework uses a dataset designed to emulate operator network architecture. It consists of TCP traffic flows captured over two days in June 2023 (07 June and 10 June) for three major service providers: **Google/YouTube**, **Facebook**, and **Deezer**.

The dataset includes the following fields:
* **starttime**: The start time of the TCP session.
* **stoptime**: The stop time of the TCP session.
* **delay_syn_synack**: The delay from a base station to the core network (CN), mapped as the **front-hauling delay**.
* **delay_synack_ack**: The delay from a base station to a mobile application server located at the CN border, mapped as the **internet edge delay**.
* **service provider**: The entity providing the content (Google, Facebook, or Deezer).
* **tac**: The Tracking Area Code of the user equipment and base station in which the TCP flows are grouped.

### Download Link
You can download the dataset here:  
[Download Dataset](https://box.roc.cnam.fr/index.php/s/M5WPWXMd5CDjpqG), we use the `donnees_topologiques_brutes.csv` file. 

---

## 🛠 Installation
This project requires **Python 3.12.8**, **NVIDIA GPU with CUDA support (for GPU acceleration)** and **At least 8GB of RAM**.

1. **Create a virtual environment and install requirements:**
   ```bash
   python -m venv env
   source env/bin/activate
   pip install -r requirements.txt
## ✍️ Citation and Contact

If you use our work, please cite the paper:


    @article{chaouche2026clinfl,
      title     = {CLIN-FL: Cost-efficient Load-balancing for In-Network Federated Learning with non-IID data},
      author    = {Chaouche, Yasmine and Ntumba, Patient and Secci, Stefano and Velloso, Pedro B.},
      journal   = {Computer Networks},
      volume    = {290},
      pages     = {112832},
      year      = {2026},
      month     = nov,
      publisher = {Elsevier},
      doi       = {10.1016/j.comnet.2026.112832}
    }


If you would like to get in touch please contact: yasmine.chaouche@lecnam.net

## 🧑‍💻 Run experiments

### 1. FL Data generation (`generate_FL_Data.py`)
This part of the framework generates federated learning datasets with broker-based architecture, topology-aware latency simulation, and data distribution analysis using Jensen-Shannon Divergence (JSD).

Edit the configuration section in generate_FL_data.py:

    dataset = 'CICIDS2017'  # or '5G3E'
    seeds = [42, 123, 456]
    nbBrokers = 3
    nbclients = 5
    nbDSPerBroker = 2
    deactRatio = 0.2  # 20% deactivation ratio
    generateConvFL = False  # True for conventional FL split

### Output files:
After running, the script generates:

    ./
    ├── FLSplit_CICIDS2017/  (if dataset == 'CICIDS2017')
    │   ├── CICIDS2017_validation.csv
    |   ├── CICIDS2017_test.csv
    │   └── JSD_Matrixes/
    │       ├── JSD_matrix_seed_42_Deact-20_CICIDS2017.pdf
    │       └── ... (one per seed)
    |
    ├── seed_<seed>_results/
    │   └── brokerBased-<deactRatio>_split/
    │       ├── AIF_1_brokerBased-<deactRatio>_split.csv
    │       ├── AIF_2_brokerBased-<deactRatio>_split.csv
    │       └── ... (one file per active client)
    │
    │
    └── JSD_matrix_<deactRatio>_CICIDS2017.pdf (aggregated results)

### Output Descriptions

- **AIF Data Files:** Individual CSV files for each active client (AIF) containing their assigned training samples
- **Validation File:** Hold-out validation dataset (20% of training data)
- **JSD Matrices:**

    - Per-seed matrices showing feature-level divergence from global distribution
    - Aggregated matrix with mean ± confidence intervals across seeds


### 2. FL training (`main.py`)

    python main.py [OPTIONS]
    
## Arguments

### Network Configuration

**`--nbclients`** 
number of clients participating in the FL simulation. 

**`--deactRatio`** The ratio of deactivated links (r) per broker.

### Dataset & Model

**`--dataset`** The dataset used: `CICIDS2017`or `5G3E`

**`--model`** The ML model used for the FL simulation: `autoencoder` or `FL-DASVDD`

### Training Configuration

**`--mode`** Specifies whether to train a new model or test an saved one. `train` or `test`

**`--nbRounds`** The number of FL rounds to execute.

**`--localEpochs`** The number of local training epochs each client performs for each FL round.

**`--batchSize`** The batch size used during local training on clients.

**`--learningRate`** The learning rate for local model training.

**`--weightDecay`** The weight decay (L2 regularization) coefficient for local training.

### FL-DASVDD Specific Parameters

**`--K`** The portion of samples used for model update in the FL-DASVDD model. 

**`--T`** The total number of initializations for the FL-DASVDD model. 

### Optimizer Configuration

**`--Adagard_lr`** The learning rate for the Adagrad optimizer.

**`--Adagard_lr_decay`** The learning rate decay factor for the Adagrad optimizer.