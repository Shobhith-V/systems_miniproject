# Andrew ID: lletitia, Name: Letitia Nimshi
import os
import numpy as np
import pandas as pd

from preprocessing import (
    load_data, split_normal_data, fit_standardization, standardize,
    CHANNEL_COLS, META_COLS,
)
from pca import fit_pca
from ridge import create_lagged_data, fit_ridge, fit_residual_std

RUN_KEYS = ["faultNumber", "simulationRun"]


def _with_meta(meta, contrib):
    # attach metadata columns to a (rows x 52) contribution matrix
    return pd.concat(
        [meta.reset_index(drop=True), pd.DataFrame(contrib, columns=CHANNEL_COLS)],
        axis=1,
    )


def spe_contributions(data, pca, k):
    # contribution of channel j = (z_j - (P P^T z)_j)^2
    Z = data[CHANNEL_COLS].to_numpy()
    P = pca.components_[:k].T                      # (52, k)
    Z_hat = (Z - pca.mean_) @ P @ P.T + pca.mean_
    return _with_meta(data[META_COLS], (Z - Z_hat) ** 2)


def ridge_contributions(data, model, lag_cols, residual_std):
    # contribution of channel j = squared standardized residual
    lagged = create_lagged_data(data)
    residual = lagged[CHANNEL_COLS].to_numpy() - model.predict(lagged[lag_cols])
    return _with_meta(lagged[META_COLS], (residual / residual_std) ** 2)


def top_channels(contrib, threshold, detector, n_top=5):
    contrib = contrib.sort_values(RUN_KEYS + ["sample"]).reset_index(drop=True)

    # the statistic is the sum of its contributions
    score = contrib[CHANNEL_COLS].sum(axis=1)

    # alarm = this sample and the two before it exceed the threshold, same run only
    exceed = (score > threshold).astype(bool)
    g = exceed.groupby([contrib["faultNumber"], contrib["simulationRun"]])
    alarm = exceed & g.shift(1, fill_value=False) & g.shift(2, fill_value=False)

    keep = alarm & (contrib["sample"] > 20)
    mean_contrib = contrib[keep].groupby("faultNumber")[CHANNEL_COLS].mean()

    rows = []
    for fault, row in mean_contrib.iterrows():
        top = row.sort_values(ascending=False).head(n_top)
        for rank, (channel, value) in enumerate(top.items(), start=1):
            rows.append({"fault": int(fault), "detector": detector,
                         "rank": rank, "channel": channel, "contribution": value})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    normal, faulty = load_data()
    train, validation, test = split_normal_data(normal)
    mean, std = fit_standardization(train)
    train = standardize(train, mean, std)
    faulty = standardize(faulty, mean, std)

    thresholds = pd.read_csv("results/thresholds.csv").set_index("detector")["threshold"]

    pca, k = fit_pca(train)
    spe = spe_contributions(faulty, pca, k)

    model, lag_cols = fit_ridge(train)
    residual_std = fit_residual_std(train, model, lag_cols)
    rdg = ridge_contributions(faulty, model, lag_cols, residual_std)

    result = pd.concat(
        [top_channels(spe, thresholds["SPE"], "SPE"),
         top_channels(rdg, thresholds["ridge"], "ridge")],
        ignore_index=True,
    )
    os.makedirs("results", exist_ok=True)
    result.to_csv("results/contributions.csv", index=False)
    print(result.groupby(["detector", "fault"]).size().unstack(0))
    print(result[result["rank"] == 1])
