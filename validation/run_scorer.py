"""End-to-end scorer harness: mirrors upstream 04's preprocessing exactly, but parameterized on the
query-features CSV so we can compare our-featurizer output vs the shipped example_features.csv, and
report test-split PCC as a scorer-sanity check. Run from the TopoDockQ repo dir.
"""
import argparse
import importlib.util
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# src/ ships a py3.8 __init__.pyc that won't import under 3.11; load the pure-source model.py directly.
_spec = importlib.util.spec_from_file_location("topodockq_model", "./src/model.py")
_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_model)
TopoDockQ = _model.TopoDockQ

PD = "./data/processed_data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", default="./feature/example_features.csv")
    ap.add_argument("--weights", default="./models/best_model.pth")
    args = ap.parse_args()

    df1 = pd.read_csv(f"{PD}/singlePPD_full_bins_features.csv")
    df_train_all = df1[df1["data_class"] != "test"]
    identical = df_train_all.columns[df_train_all.nunique() == 1]
    print(f"dropped {len(identical)} identical (training-constant) columns")
    df1 = df1.drop(columns=identical)

    df2 = pd.read_csv(f"{PD}/singlePPD_DockQ.csv")[
        ["pdb_id", "af_model_id", "af_confidence", "pdb2sql_DockQ", "data_class"]]
    df1 = pd.merge(df1, df2, on=["pdb_id", "af_model_id", "pdb2sql_DockQ", "data_class"], how="inner")

    df3 = pd.read_csv(args.features_csv)
    X_single = df3.copy()
    X_single["pdb2sql_DockQ"] = 1.0

    df_train = df1[df1["data_class"] == "train"]
    df_test = df1[df1["data_class"] == "test"]

    filtration_values = [f"{x:.1f}" if x.is_integer() else f"{x:.2f}".rstrip("0")
                         for x in np.arange(2.0, 10.25, 0.25)]
    persistent = []
    for f in filtration_values:
        persistent += [f"persistent_{f}_{str(i + 1).zfill(2)}" for i in range(72)]
    static = [f"static_{str(i + 1).zfill(2)}" for i in range(378)]
    feature_name_list = ["pdb2sql_DockQ"] + persistent + static
    valid = [c for c in feature_name_list if c in df_train.columns]

    df_train, df_test, X_single = df_train[valid], df_test[valid], X_single[valid]

    def xy(df):
        return df.drop(columns=["pdb2sql_DockQ"]).values, df["pdb2sql_DockQ"].values

    X_train, _ = xy(df_train)
    X_test, y_test = xy(df_test)
    X_s, _ = xy(X_single)
    print(f"input_dim={X_train.shape[1]}  train={X_train.shape[0]} test={X_test.shape[0]}")

    scaler = StandardScaler().fit(X_train)
    X_test = scaler.transform(X_test)
    X_s = scaler.transform(X_s)

    model = TopoDockQ(X_train.shape[1], 2048, 2048, 2048, 2048, 0.0)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        pred_single = model(torch.tensor(X_s, dtype=torch.float32)).numpy().flatten()
        pred_test = model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()

    pcc = np.corrcoef(pred_test, y_test)[0, 1]
    print(f"TEST p-DockQ vs pdb2sql_DockQ  PCC={pcc:.4f}  (n={len(y_test)})")
    print(f"QUERY predicted p-DockQ ({args.features_csv}): {pred_single}")


if __name__ == "__main__":
    main()
