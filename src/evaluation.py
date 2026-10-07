import os
import numpy as np
import pandas as pd

from preprocessing import META_COLS

RUN_KEYS = ["faultNumber", "simulationRun"]
ONSET = 20                # the fault enters after sample 20
MINUTES_PER_SAMPLE = 3
IN_A_ROW = 3              # alarm = this sample and the two before it exceed the threshold


def load_scores():
    # one table per statistic, each with the metadata columns and a "score" column
    pca = pd.read_parquet("results/scores_pca.parquet")
    ridge = pd.read_parquet("results/scores_ridge.parquet")
    return {
        "T2": pca[META_COLS + ["T2"]].rename(columns={"T2": "score"}),
        "SPE": pca[META_COLS + ["SPE"]].rename(columns={"SPE": "score"}),
        "ridge": ridge[META_COLS + ["score"]],
    }


def compute_threshold(scores):
    # 99th percentile of the scores on the validation runs (fault-free runs 301 to 400)
    validation = scores[(scores["faultNumber"] == 0) & scores["simulationRun"].between(301, 400)]
    return float(np.quantile(validation["score"].to_numpy(), 0.99))


def add_alarms(scores, threshold):
    scores = scores.sort_values(RUN_KEYS + ["sample"]).reset_index(drop=True)
    exceed = scores["score"] > threshold

    # shifting inside each run, so an alarm never crosses a run boundary
    g = exceed.groupby([scores["faultNumber"], scores["simulationRun"]])
    alarm = exceed.copy()
    for lag in range(1, IN_A_ROW):
        alarm = alarm & g.shift(lag, fill_value=False)

    scores["alarm"] = alarm
    return scores


def false_alarm_rate(scores):
    # share of samples in alarm over the test runs (fault-free runs 401 to 500)
    test = scores[(scores["faultNumber"] == 0) & scores["simulationRun"].between(401, 500)]
    return test.groupby("simulationRun")["alarm"].mean().mean()


def fault_metrics(scores, fault):
    # only the samples after the fault has entered count
    post = scores[(scores["faultNumber"] == fault) & (scores["sample"] > ONSET)]

    # detection rate of each run, then averaged over the runs
    detection_rate = post.groupby("simulationRun")["alarm"].mean().mean()

    # first alarm of each run, runs that never alarmed drop out here
    first_alarm = post[post["alarm"]].groupby("simulationRun")["sample"].min()
    delays = (first_alarm - ONSET) * MINUTES_PER_SAMPLE
    median_delay = delays.median() if len(delays) > 0 else np.nan
    runs_missed = post["simulationRun"].nunique() - len(first_alarm)

    return detection_rate, median_delay, runs_missed


def detection_table(scores, detector):
    # fault 0 is the false-alarm rate, it has no delay and no missed runs
    rows = [{"fault": 0, "detector": detector,
             "detection_rate": false_alarm_rate(scores),
             "median_delay_min": np.nan, "runs_missed": pd.NA}]

    for fault in range(1, 21):
        detection_rate, median_delay, runs_missed = fault_metrics(scores, fault)
        rows.append({"fault": fault, "detector": detector,
                     "detection_rate": detection_rate,
                     "median_delay_min": median_delay, "runs_missed": runs_missed})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_scores = load_scores()

    threshold_rows = []
    tables = []
    for detector, scores in all_scores.items():
        threshold = compute_threshold(scores)
        threshold_rows.append({"detector": detector, "threshold": threshold})

        scores = add_alarms(scores, threshold)
        tables.append(detection_table(scores, detector))

    thresholds = pd.DataFrame(threshold_rows)
    detection = pd.concat(tables, ignore_index=True)
    detection["runs_missed"] = detection["runs_missed"].astype("Int64")

    os.makedirs("results", exist_ok=True)
    thresholds.to_csv("results/thresholds.csv", index=False)
    detection.to_csv("results/detection.csv", index=False)

    print(thresholds)
    print(detection.pivot(index="fault", columns="detector", values="detection_rate").round(4))
    print(detection.pivot(index="fault", columns="detector", values="median_delay_min"))
    print(detection.pivot(index="fault", columns="detector", values="runs_missed"))
