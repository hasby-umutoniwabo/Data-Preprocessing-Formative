"""
Task 1: merge customer_social_profiles + customer_transactions into one dataset.
Social IDs look like "A178", transaction IDs look like "151" - same customers,
different format, so we strip the "A" and match them as plain numbers.
Social profiles also have multiple rows per customer (one per platform), so
we average/aggregate those down to one row per customer before merging.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SOCIAL_CSV = "data/raw/customer_social_profiles.csv"
TRANSACTIONS_CSV = "data/raw/customer_transactions.csv"
SOCIAL_ID_COL = "customer_id_new"
TRANSACTIONS_ID_COL = "customer_id_legacy"
TARGET_COL = "product_category"  # what the recommendation model will predict
OUTPUT_CSV = "data/processed/merged_dataset.csv"
EDA_PLOTS_DIR = "data/processed/eda_plots"


def load_data():
    # Read both raw CSVs into DataFrames
    social = pd.read_csv(SOCIAL_CSV)
    transactions = pd.read_csv(TRANSACTIONS_CSV)
    return social, transactions


def run_eda(social, transactions):
    # Summary stats + dtypes for both datasets, printed for the report
    os.makedirs(EDA_PLOTS_DIR, exist_ok=True)
    print("=== Social profiles: dtypes ===")
    print(social.dtypes)
    print("\n=== Social profiles: summary stats ===")
    print(social.describe(include="all"))
    print("\n=== Transactions: dtypes ===")
    print(transactions.dtypes)
    print("\n=== Transactions: summary stats ===")
    print(transactions.describe(include="all"))

    # Plot 1: how purchase amounts are distributed
    plt.figure(figsize=(6, 4))
    sns.histplot(transactions["purchase_amount"], bins=20, kde=True)
    plt.title("Distribution of Purchase Amount")
    plt.tight_layout()
    plt.savefig(f"{EDA_PLOTS_DIR}/dist_purchase_amount.png", dpi=110)
    plt.close()

    # Plot 2: check for outlier purchase amounts
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=transactions["purchase_amount"])
    plt.title("Outliers: Purchase Amount")
    plt.tight_layout()
    plt.savefig(f"{EDA_PLOTS_DIR}/outliers_purchase_amount.png", dpi=110)
    plt.close()

    # Plot 3: how numeric social features relate to each other
    numeric_social = social.select_dtypes(include="number")
    plt.figure(figsize=(5, 4))
    sns.heatmap(numeric_social.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation: Social Profile Numeric Features")
    plt.tight_layout()
    plt.savefig(f"{EDA_PLOTS_DIR}/correlation_social.png", dpi=110)
    plt.close()

    # Plot 4: is the target balanced across product categories?
    plt.figure(figsize=(6, 4))
    transactions[TARGET_COL].value_counts().plot(kind="bar")
    plt.title(f"Class balance: {TARGET_COL}")
    plt.tight_layout()
    plt.savefig(f"{EDA_PLOTS_DIR}/target_class_balance.png", dpi=110)
    plt.close()
    print(f"\nSaved 4 labeled EDA plots -> {EDA_PLOTS_DIR}/")


def clean_social(social):
    # Drop exact duplicate rows
    social = social.copy()
    before = len(social)
    social = social.drop_duplicates()
    print(f"Dropped {before - len(social)} exact duplicate social profile row(s)")

    # Normalize "A178" -> 178 so it can be matched against transactions later
    social["customer_id"] = social[SOCIAL_ID_COL].str.replace("A", "", regex=False).astype(int)

    # Each customer has multiple rows (one per platform) - collapse to one row per customer
    agg_social = social.groupby("customer_id").agg(
        engagement_score=("engagement_score", "mean"),
        purchase_interest_score=("purchase_interest_score", "mean"),
        social_media_platform=("social_media_platform", lambda s: s.mode()[0]),
        review_sentiment=("review_sentiment", lambda s: s.mode()[0]),
    ).reset_index()
    print(f"Aggregated social profiles: {len(social)} rows -> {len(agg_social)} customers")
    return agg_social


def clean_transactions(transactions):
    # Drop duplicate transaction rows
    transactions = transactions.copy()
    before = len(transactions)
    transactions = transactions.drop_duplicates()
    print(f"Dropped {before - len(transactions)} duplicate transaction row(s)")

    # Rename ID column to match social profiles, fix the date type
    transactions = transactions.rename(columns={TRANSACTIONS_ID_COL: "customer_id"})
    transactions["purchase_date"] = pd.to_datetime(transactions["purchase_date"])

    # Fill missing ratings with the median rating
    n_missing = transactions["customer_rating"].isna().sum()
    median_rating = transactions["customer_rating"].median()
    transactions["customer_rating"] = transactions["customer_rating"].fillna(median_rating)
    print(f"Filled {n_missing} missing customer_rating value(s) with median ({median_rating})")
    return transactions


def merge_and_validate(social, transactions):
    # Inner join on customer_id - only keep transactions with a matching social profile
    merged = pd.merge(transactions, social, on="customer_id", how="inner")

    # Sanity-check the merge worked as expected
    print("\n=== Merge validation ===")
    print(f"Social customers      : {social['customer_id'].nunique()}")
    print(f"Transaction rows      : {len(transactions)}")
    print(f"Merged rows           : {len(merged)}")
    unmatched = set(transactions["customer_id"]) - set(social["customer_id"])
    print(f"Transactions with no matching social profile (dropped): {len(unmatched)}")
    nulls = merged.isna().sum()
    print(f"Nulls remaining after merge:\n{nulls[nulls > 0] if nulls.any() else 'none'}")
    return merged


def engineer_features(merged):
    # Add a couple of derived features that might help the model
    merged = merged.copy()
    merged["days_since_purchase"] = (pd.Timestamp.today() - merged["purchase_date"]).dt.days
    merged["engagement_x_interest"] = merged["engagement_score"] * merged["purchase_interest_score"]
    return merged


def main():
    social, transactions = load_data()
    run_eda(social, transactions)

    social_clean = clean_social(social)
    transactions_clean = clean_transactions(transactions)

    merged = merge_and_validate(social_clean, transactions_clean)
    merged = engineer_features(merged)

    # Save the final merged + engineered dataset
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved merged dataset ({merged.shape[0]} rows, {merged.shape[1]} cols) -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()