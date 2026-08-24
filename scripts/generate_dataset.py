"""
generate_dataset.py
--------------------
Expands the authentic IBM/Kaggle Telco Customer Churn seed sample
(data/telco_seed_real.csv - 367 real customer records fetched from the
public dataset mirror) into a larger training set, WITHOUT leaking
near-duplicate rows across the eventual train/test split.

Method:
  1. Split the 367 *real* customers into train/test seed groups first
     (stratified by Churn), before any augmentation.
  2. Independently bootstrap-resample + jitter numeric fields within
     each group. Because a jittered "clone" of a real customer can only
     land in the same group its original came from, the train and test
     sets never share near-duplicate rows.

This preserves every real joint categorical relationship (Fiber optic +
Month-to-month + high charges -> higher churn, etc.) since we resample
whole rows, while giving an honest, leak-free evaluation set.

Run:
    python generate_dataset.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RNG = np.random.default_rng(42)
N_TRAIN_TARGET = 3200
N_TEST_TARGET = 800

SEED_PATH = "../data/telco_seed_real.csv"
OUT_TRAIN_PATH = "../data/telco_churn_train.csv"
OUT_TEST_PATH = "../data/telco_churn_test.csv"
OUT_FULL_PATH = "../data/telco_churn.csv"  # combined, for dashboard stats


def jitter_row(row, i, tag):
    row = row.copy()
    tenure = int(np.clip(row["tenure"] + RNG.integers(-4, 5), 0, 72))
    row["tenure"] = tenure

    monthly = float(np.clip(row["MonthlyCharges"] + RNG.normal(0, 2.5), 18.0, 120.0))
    monthly = round(monthly, 2)
    row["MonthlyCharges"] = monthly

    base_total = monthly * max(tenure, 0)
    noise = RNG.normal(0, base_total * 0.03 + 1)
    total = round(max(base_total + noise, monthly if tenure > 0 else 0), 2)
    row["TotalCharges"] = total

    row["customerID"] = f"{RNG.integers(1000, 9999)}-{tag}{i:05d}"
    return row


def expand(seed_subset, target_n, tag):
    n_extra = max(target_n - len(seed_subset), 0)
    idx = RNG.integers(0, len(seed_subset), size=n_extra)
    extra_rows = [jitter_row(seed_subset.iloc[j], i, tag) for i, j in enumerate(idx)]
    extra_df = pd.DataFrame(extra_rows)
    out = pd.concat([seed_subset, extra_df], ignore_index=True)
    return out.sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    seed = pd.read_csv(SEED_PATH)
    seed["TotalCharges"] = pd.to_numeric(seed["TotalCharges"], errors="coerce")
    seed = seed.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    print(f"Loaded {len(seed)} real seed rows "
          f"(churn rate: {(seed['Churn'] == 'Yes').mean():.1%})")

    seed_train, seed_test = train_test_split(
        seed, test_size=0.2, random_state=42, stratify=seed["Churn"]
    )
    print(f"Real seed split -> train customers: {len(seed_train)}, "
          f"test customers: {len(seed_test)} (split BEFORE augmentation)")

    train_df = expand(seed_train, N_TRAIN_TARGET, "TR")
    test_df = expand(seed_test, N_TEST_TARGET, "TE")

    train_df.to_csv(OUT_TRAIN_PATH, index=False)
    test_df.to_csv(OUT_TEST_PATH, index=False)

    full = pd.concat([train_df, test_df], ignore_index=True)
    full.to_csv(OUT_FULL_PATH, index=False)

    print(f"Wrote {len(train_df)} rows -> {OUT_TRAIN_PATH} "
          f"(churn rate {(train_df['Churn'] == 'Yes').mean():.1%})")
    print(f"Wrote {len(test_df)} rows -> {OUT_TEST_PATH} "
          f"(churn rate {(test_df['Churn'] == 'Yes').mean():.1%})")
    print(f"Wrote {len(full)} combined rows -> {OUT_FULL_PATH}")


if __name__ == "__main__":
    main()
