# Feature Skew Control for In-Network Federated Learning

[![Python Version](https://img.shields.io/badge/python-3.12.8-blue)](https://www.python.org/downloads/release/python-3128/)

This repository provides the official implementation of the framework presented in:  
**"Feature Skew Control for In-Network Federated Learning"** *Yasmine Chaouche, Patient Ntumba, Stefano Secci, Pedro B. Velloso (Cnam, Paris, France)*

## 📄 Abstract

---

## 📊 Datasets
We evaluate our framework using two major network traffic datasets:

1.  **5G3Evl (5G End-to-End Emulation):** Time-series data from a 5G network testbed covering radio, computing, and network resources. It includes anomalies like CPU overload, link failures, and packet loss[cite: 585, 587].
2.  **CIC-IDS-2017:** A comprehensive dataset for evaluating intrusion detection systems, featuring real-world network traffic with attacks like DDoS, SQL Injection, and Brute Force.


---

## 🛠 Installation
This project requires **Python 3.12.8**.

1. **Create a virtual environment and install requirements:**
   ```bash
   python -m venv env
   source env/bin/activate
   pip install -r requirements.txt
## ✍️ Citation and Contact

If you use our work, please also site the paper:


    @article{chaouche2024feature,
    title={Feature Skew Control for In-Network Federated Learning},
    author={Chaouche Yasmine, Ntumba Patient, Secci Stefano and Velloso Pedro B.},
    journal={X Journal (Name of the Journal)},
    year={2025},
    publisher={..}
    }


If you would like to get in touch please contact: yasmine.chaouche@lecnam.net