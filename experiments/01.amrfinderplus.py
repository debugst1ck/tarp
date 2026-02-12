# %%
from pathlib import Path

import polars as pl

# temp\data\raw\reference_gene_catalog.20260112.txt tsv file columns

rcg_path = Path("temp/data/raw/reference_gene_catalog.20260112.txt")

rcg_df = pl.read_csv(rcg_path, separator="\t")

print(rcg_df.head())
