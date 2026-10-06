import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge

from preprocessing import (
    load_data,
    split_normal_data,
    fit_standardization,
    standardize,
    CHANNEL_COLS,
)
def create_lagged_data(data):
    # creating lagged features for each channel
    lagged_data = data.copy()
    for col in CHANNEL_COLS:
        lagged_data[f"{col}_lag1"] = (lagged_data.groupby(["faultNumber","simulationRun"])[col].shift(1))
        lagged_data[f"{col}_lag2"] = (lagged_data.groupby(["faultNumber","simulationRun"])[col].shift(2))
    lagged_data = lagged_data.dropna().reset_index(drop=True)

    return lagged_data

def fit_ridge(train):
    lagged_train=create_lagged_data(train)
    lag_cols=[f"{col}_lag1" for col in CHANNEL_COLS] + [f"{col}_lag2" for col in CHANNEL_COLS]
    X_train = lagged_train[lag_cols]
    y_train = lagged_train[CHANNEL_COLS]
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model, lag_cols

def fit_residual_std(train, model, lag_cols):
    lagged_train = create_lagged_data(train)
    X_train = lagged_train[lag_cols]
    y_train = lagged_train[CHANNEL_COLS]
    prediction = model.predict(X_train)
    residual = y_train.to_numpy() - prediction
    residual_std = np.std(residual, axis=0, ddof=1)
    return residual_std

def calculate_ridge_scores(data, model, lag_cols, residual_std):
    lagged_data = create_lagged_data(data)
    X = lagged_data[lag_cols]
    y = lagged_data[CHANNEL_COLS]

    prediction = model.predict(X)
    residual = y.to_numpy() - prediction
    score = np.sum((residual / residual_std) ** 2,axis=1)
    results = lagged_data[["faultNumber", "simulationRun", "sample"]].copy()
    results["score"] = score

    return results

if __name__ == "__main__":
    # Load data
    normal, faulty = load_data()
    train, validation, test = split_normal_data(normal)
    mean, std = fit_standardization(train)

    train = standardize(train, mean, std)
    validation = standardize(validation, mean, std)
    test = standardize(test, mean, std)
    faulty = standardize(faulty, mean, std)

    model, lag_cols = fit_ridge(train)

    residual_std = fit_residual_std(train,model,lag_cols)
    print("Residual std shape:", residual_std.shape)

    validation_results = calculate_ridge_scores(validation,model,lag_cols,residual_std)
    test_results = calculate_ridge_scores(test,model,lag_cols,residual_std)
    faulty_results = calculate_ridge_scores(faulty, model, lag_cols, residual_std)

    # Combine results
    results = pd.concat([validation_results,test_results,faulty_results],ignore_index=True)
    results.to_parquet("results/scores_ridge.parquet",index=False)
    print(f"Saved {len(results)} Ridge scores")
    print(results.head())

    print("Validation:", len(validation_results))
    print("Test:", len(test_results))
    print("Faulty:", len(faulty_results))
    print("Validation runs:",validation_results["simulationRun"].nunique())
    print("Test runs:", test_results["simulationRun"].nunique())
    print("Faulty runs:", faulty_results["simulationRun"].nunique())