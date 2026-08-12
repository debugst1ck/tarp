from pathlib import Path

import numpy as np
import polars as pl
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tqdm.auto import tqdm

from tarp.data.sources.sequence import GenomeSliceSource, TabularSequenceSource

pl.Config.set_tbl_rows(100)

dna_source = GenomeSliceSource(
    genomes_directory=Path("temp/data/external/sequences/nucleotides"),
    metadata_source=Path("temp/data/interim/bacteria.gene.fine_tuning.parquet"),
    key_column="genomic_nucleotide_accession.version",
    start_column="start_position_on_the_genomic_accession",
    end_column="end_position_on_the_genomic_accession",
    sequence_column="dna_sequence",
) + TabularSequenceSource(
    source=Path("temp/data/interim/card_amr.parquet"),
)

metadata_paths = [
    Path("temp/data/interim/bacteria.gene.fine_tuning.parquet"),
    Path("temp/data/interim/card_amr.parquet"),
]

label_columns = pl.read_csv(Path("temp/data/cache/labels.csv")).to_series().to_list()

# Grab indices based on height and whether it's from card_amr or not
indices = np.array(range(dna_source.height))

# Since its a multilabel classification problem, we need to stratify based on the labels.
# Count if the label is present or not for each row,

labels = []

for index in tqdm(range(dna_source.height)):
    row = dna_source.retrieve(index)
    label = np.array([row.get(col, 0) for col in label_columns])
    labels.append(label)

labels = np.vstack(labels)

sum = labels.sum(axis=0)

common_columns = np.where(sum > 10)[0]

reduced_label_columns = [label_columns[i] for i in common_columns]

# 80 / 10 / 10 split

# Get good indices = rows where at least one of the good labels is present
reduced_indices = np.where(labels[:, common_columns].sum(axis=1) > 0)[0]

stratifier = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=69420)

# Stratifier yields 10 splits (each temp_indices is ~10% of data)
folds = list(
    stratifier.split(reduced_indices, labels[reduced_indices][:, common_columns])
)

folds_indices = []
for _, test_idx in stratifier.split(
    reduced_indices, labels[reduced_indices][:, common_columns]
):
    folds_indices.append(reduced_indices[test_idx])

five_folds = []
for k in range(5):
    val_idx = folds_indices[2 * k]
    test_idx = folds_indices[2 * k + 1]

    # Train is concatenation of remaining 8 folds
    train_idx_list = [
        folds_indices[i] for i in range(10) if i != 2 * k and i != 2 * k + 1
    ]
    train_idx = np.concatenate(train_idx_list)

    five_folds.append((train_idx, val_idx, test_idx))

# Iterate though dna_source
global_offsets = dna_source.cumulative_heights.numpy()

for i, source in tqdm(enumerate(dna_source.sources), total=len(dna_source.sources)):
    metadata_table = pl.read_parquet(metadata_paths[i])
    name = metadata_paths[i].name

    start_offset = global_offsets[i]
    end_offset = global_offsets[i + 1]

    for fold, (train_idx, val_idx, test_idx) in enumerate(five_folds):
        local_train = (
            train_idx[(train_idx >= start_offset) & (train_idx < end_offset)]
            - start_offset
        )
        local_val = (
            val_idx[(val_idx >= start_offset) & (val_idx < end_offset)] - start_offset
        )
        local_test = (
            test_idx[(test_idx >= start_offset) & (test_idx < end_offset)]
            - start_offset
        )

        # Positional selection via Polars expression (No 'index' column required)
        train_metadata = metadata_table[local_train]
        val_metadata = metadata_table[local_val]
        test_metadata = metadata_table[local_test]

        # Directory handling and writing
        for split_name, df in [
            ("train", train_metadata),
            ("val", val_metadata),
            ("test", test_metadata),
        ]:
            out_path = Path(
                f"temp/data/processed/finetuning/fold_{fold}/{split_name}/{name}"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(out_path)

# Write the reduced label columns to a csv for later use
pl.DataFrame({"label": reduced_label_columns}).write_csv(
    Path("temp/data/cache/labels_reduced.csv")
)
