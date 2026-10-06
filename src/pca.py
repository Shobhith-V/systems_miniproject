import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from preprocessing import (
    load_data,
    split_normal_data,
    fit_standardization,
    standardize,
    CHANNEL_COLS,
)
def fit_pca(train):
    # extracting the features for PCA and finding the number of components to retain 90% variance
    X_train = train[CHANNEL_COLS]
    pca = PCA()
    pca.fit(X_train)
    k = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.90) + 1
    return pca,k

def calculate_scores(data,pca,k):
    # we need T^2 and SPE
    X = data[CHANNEL_COLS]
    X_pca = pca.transform(X)
    T2 = np.sum((X_pca[:, :k] / np.sqrt(pca.explained_variance_[:k]))**2, axis=1)
    X_reconstructed = X_pca[:, :k] @ pca.components_[:k, :]
    SPE = np.sum((X - X_reconstructed)**2, axis=1)
    return T2, SPE

def make_results(data, pca, k):
    T2, SPE = calculate_scores(data, pca, k)
    results = data[["faultNumber", "simulationRun", "sample"]].copy()
    results["T2"] = T2
    results["SPE"] = SPE

    return results


if __name__ == "__main__":
    # Load raw data
    normal, faulty = load_data()

    # Splitting normal data into train/validation/test
    train, validation, test = split_normal_data(normal)
    # Fit standardization using training data only
    mean, std = fit_standardization(train)

    # Standardize all datasets using training statistics
    train = standardize(train, mean, std)
    validation = standardize(validation, mean, std)
    test = standardize(test, mean, std)
    faulty = standardize(faulty, mean, std)

    # Fit PCA using training data only
    pca, k = fit_pca(train)

    print(f"Number of retained components: {k}")
    print(f"Explained variance: "f"{np.sum(pca.explained_variance_ratio_[:k]):.4f}")

    # Calculating PCA scores
    validation_results = make_results(validation, pca, k)
    test_results = make_results(test, pca, k)
    faulty_results = make_results(faulty, pca, k)

    # Combining results
    results = pd.concat([validation_results, test_results, faulty_results],ignore_index=True)
    
    # Save results
    results.to_parquet("results/scores_pca.parquet",index=False)

    print(f"Saved {len(results)} PCA scores")
    print(results.head())