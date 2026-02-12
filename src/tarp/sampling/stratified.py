from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import KFold, StratifiedKFold

pl.Config.set_tbl_rows(-1)

paths = [
    Path("temp/data/interim/card_amr.parquet"),
    Path("temp/data/interim/bacteria_gene_fine_tuning.parquet"),
]

datasets: list[pl.DataFrame] = [pl.read_parquet(path) for path in paths]

column_names: list[str] = (
    pl.read_csv(Path("temp/data/cache/labels.csv")).to_series().to_list()
)

# We can merge columns into broader categories if needed
# For example, merging all antibiotic columns into a single 'amr' category
merge_columns = {
    "amr": [column for column in column_names if column != "non_amr"],
    "non_amr": ["non_amr"],
}

# We need to split each dataset into train, valid, test sets 80/10/10
# While maintaining the distribution of amr/non_amr labels using stratified sampling
# We also need to ensure that each split preserves
# Although stratification can be done global indices, it needs to be split back to each dataset and local indices
# So we will first create a global stratified split, then map back to local indices

# Lets get the required columns from each dataset
labels = []
for dataset in datasets:
    # Extract columns of interest
    # In case columns are missing in some datasets, we can fill them with zeros
    missing = [c for c in column_names if c not in dataset.columns]
    if missing:
        ds = dataset.with_columns([pl.lit(0).alias(c) for c in missing]).select(
            column_names
        )
    else:
        ds = dataset.select(column_names)

    # Ensure dtype of int 8
    ds = ds.with_columns([pl.col(c).cast(pl.Int8) for c in column_names])

    labels.append(ds)

sss = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create a combined stratification label
# We will create a single label by merging columns as per merge_columns

combined_labels = []
for label_df in labels:
    expressions = []
    for target, columns in merge_columns.items():
        expr = pl.any_horizontal(pl.col(columns)).cast(pl.Int8).alias(target)
        expressions.append(expr)

    combined = label_df.select(expressions)

    combined_labels.append(combined)

# Concatenate all combined labels
all_labels = pl.concat(combined_labels, how="vertical")

all_fine_labels = pl.concat(labels, how="vertical")

print(all_labels.select(pl.all().sum()) / all_labels.height)


def compare_label_distributions(train_df: pl.DataFrame, test_df: pl.DataFrame):
    train_prop = (train_df.select(pl.all().mean())).transpose(
        include_header=True, header_name="label", column_names=["train"]
    )

    test_prop = (test_df.select(pl.all().mean())).transpose(
        include_header=True, header_name="label", column_names=["test"]
    )

    comparison = (
        train_prop.join(test_prop, on="label")
        .with_columns((pl.col("train") - pl.col("test")).abs().alias("abs_diff"))
        .sort("abs_diff", descending=True)
    )

    return comparison


for i, (train_index, test_index) in enumerate(
    sss.split(np.zeros(all_labels.height), all_labels.to_numpy().argmax(axis=1))
):
    print(f"Fold {i}:")

    # Print distribution in train and test sets
    train_labels = all_labels[train_index]
    test_labels = all_labels[test_index]

    train_fine = all_fine_labels[train_index]
    test_fine = all_fine_labels[test_index]

    comparison = compare_label_distributions(train_fine, test_fine)

    # Print sizes too
    print(f"Train size: {train_labels.height}, Test size: {test_labels.height}")
    print(comparison)

    # Now we need to map back to local indices for each dataset
    current_index = 0
    for ds_id, dataset in enumerate(datasets):
        ds_size = dataset.height

        # Get local train and test indices
        local_train_indices = [
            idx - current_index
            for idx in train_index
            if current_index <= idx < current_index + ds_size
        ]
        local_test_indices = [
            idx - current_index
            for idx in test_index
            if current_index <= idx < current_index + ds_size
        ]

        # Create train and test splits
        train_split = dataset[local_train_indices]
        test_split = dataset[local_test_indices]

        # Save splits
        train_path = Path(
            f"temp/data/processed/finetuning/fold_{i}/train/{paths[ds_id].name}"
        )
        test_path = Path(
            f"temp/data/processed/finetuning/fold_{i}/test/{paths[ds_id].name}"
        )
        train_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.parent.mkdir(parents=True, exist_ok=True)

        train_split.write_parquet(train_path)
        test_split.write_parquet(test_path)

        print(
            f"Dataset {ds_id} Fold {i}: Train size: {train_split.height}, Test size: {test_split.height}"
        )

        current_index += ds_size
