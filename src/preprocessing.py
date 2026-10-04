# first we need to seperate the metadata from the actual plant measurements
import pandas as pd
import numpy as np

META_COLS=["faultNumber","simulationRun","sample"]

CHANNEL_COLS=([f"xmeas_{i}" for i in range(1, 42)]+[f"xmv_{i}" for i in range(1, 12)])

def load_data():
    normal=pd.read_parquet("data/tep_fault_free_training.parquet")
    faulty=pd.read_parquet("data/tep_faulty_training_runs01-20.parquet")
    return normal, faulty

def split_normal_data(normal):
    train = normal[normal["simulationRun"].between(1, 300)].copy()
    validation = normal[normal["simulationRun"].between(301, 400)].copy()
    test = normal[normal["simulationRun"].between(401, 500)].copy()

    return train, validation, test


def fit_standardization(train):
    mean = train[CHANNEL_COLS].mean()
    std = train[CHANNEL_COLS].std(ddof=1)

    return mean, std


def standardize(data, mean, std):
    result = data.copy()
    result[CHANNEL_COLS] = (result[CHANNEL_COLS] - mean) / std

    return result
