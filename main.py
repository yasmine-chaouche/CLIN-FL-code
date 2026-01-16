import argparse
import utils
import pandas as pd
import numpy as np
import cudf as cdf
from sklearn.cluster import KMeans
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from sklearn.preprocessing import MinMaxScaler
import generate_FL_Data

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--nbclients',type=int,default=5,help="Number of clients")
    parser.add_argument('--nbBrokers',type=int,default=5,help="Number of brokers")
    parser.add_argument('--nbDSPerBroker',type=int,default=1,help="Number of datasources per broker")
    parser.add_argument('--dataset',type=str,default='CICIDS2017',help="dataset used for FL simulation")
    parser.add_argument('--model',type=str,default='autoencoder',help="model used for FL simulation: autoencoder or DASVDD")
    parser.add_argument('--threshold',type=float,default=0.09,help="Variance threshold for feature selection")
    parser.add_argument('--deactRatio',type=float,default=0.8,help="Ratio of deactivated links per broker")
    parser.add_argument('--generateConvFL',action='store_true',help="Whether to generate conventional FL split or broker-based split")
    args=parser.parse_args()
    return args


if __name__=='__main__':
    args=get_args()
