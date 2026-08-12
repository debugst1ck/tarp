import os
import ssl
import time
import urllib.request
from pathlib import Path

import certifi
import polars as pl
from Bio import Entrez, SeqIO
from tqdm.auto import tqdm


def download_sequences_bulk(
    lf: pl.LazyFrame, target_col: str, db: str, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract unique accession identifiers
    accessions = lf.select(target_col).unique().collect().to_series().to_list()
    existing = {f.stem for f in output_dir.glob("*.fasta")}

    # Check if NZ_CP013742.1 is in the list of accessions
    todo = [acc for acc in accessions if acc not in existing]

    if not todo:
        print(f"All target sequences in {db} database are accounted for.")
        return

    # Maximize allowable fetch chunk sizing (NCBI maximum is ~500 IDs per request via POST)
    batch_size = 100
    delay = (
        0.15 if Entrez.api_key else 0.4
    )  # Dynamically conform to NCBI rate-limiting parameters

    print(f"Downloading {len(todo)} accessions from NCBI [{db}] database...")

    for i in tqdm(range(0, len(todo), batch_size)):
        batch = todo[i : i + batch_size]

        try:
            # epost stores accessions on NCBI server temporarily to prevent URL character length truncation
            search_handle = Entrez.epost(db=db, id=",".join(batch))
            search_results = Entrez.read(search_handle)
            webenv = search_results["WebEnv"]
            query_key = search_results["QueryKey"]
            search_handle.close()

            # Retrieve consolidated record mapping using Web History environment parameters
            fetch_handle = Entrez.efetch(
                db=db,
                rettype="fasta",
                retmode="text",
                webenv=webenv,
                query_key=query_key,
            )

            records = list(SeqIO.parse(fetch_handle, "fasta"))
            fetch_handle.close()

            # Vectorized local storage allocation
            for record in records:
                # Sanitize sequence IDs to isolate specific base accessions
                accession_id = (
                    record.id if record.id is not None else ("unknown_accession")
                )
                output_path = output_dir / f"{accession_id}.fasta"
                SeqIO.write(record, output_path, "fasta")

            time.sleep(delay)

        except Exception as e:
            print(
                f"\nExecution anomaly during index range [{i}:{i + batch_size}] on database [{db}]: {e}"
            )
            time.sleep(2.0)  # Extended cooling-off phase prior to reconnection retry
            continue


def main() -> None:
    pl.set_random_seed(42)

    pre_training_count = 1_000_000
    fine_tuning_count = 6385

    # bacteria_gene_info = pl.scan_parquet(
    #     Path("temp/data/raw/bacteria.gene_info.20251001.parquet")
    # )
    # gene2accession = pl.scan_parquet(
    #     Path("temp/data/raw/gene2accession.20251006.parquet")
    # )
    # protein_coding_genes = (
    #     bacteria_gene_info.filter(pl.col("type_of_gene") == "protein-coding")
    #     .select("GeneID")
    #     .join(gene2accession, on="GeneID", how="inner")
    #     .filter(
    #         pl.col("protein_accession.version").is_not_null()
    #         & pl.col("genomic_nucleotide_accession.version").is_not_null()
    #     )
    #     .filter(
    #         pl.col("end_position_on_the_genomic_accession").is_not_null()
    #         & pl.col("start_position_on_the_genomic_accession").is_not_null()
    #     )
    #     .filter(
    #         pl.col("end_position_on_the_genomic_accession")
    #         != pl.col("start_position_on_the_genomic_accession")
    #     )
    # )
    # sample = protein_coding_genes.limit(pre_training_count + fine_tuning_count)
    # sample.select(
    #     [
    #         "GeneID",
    #         "#tax_id",
    #         "status",
    #         "genomic_nucleotide_accession.version",
    #         "genomic_nucleotide_gi",
    #         "protein_gi",
    #         "protein_accession.version",
    #         "start_position_on_the_genomic_accession",
    #         "end_position_on_the_genomic_accession",
    #         "orientation",
    #         "Symbol",
    #     ]
    # ).with_columns(pl.lit(1).alias("non_amr")).sink_parquet(
    #     Path("temp/data/interim/bacteria.genes.parquet")
    # )

    # Use API key to increase rate limit from 3 to 10 requests per second
    Entrez.email = os.getenv("NCBI_REGISTERED_EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")  # Set this in your environment variables

    # Configure SSL context safely
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl_context)
    )
    urllib.request.install_opener(opener)

    # Scan dataset
    non_amr = pl.scan_parquet(Path("temp/data/interim/bacteria.genes.parquet"))

    # Processing pipelines for Nucleotides and Proteins
    download_sequences_bulk(
        non_amr,
        "genomic_nucleotide_accession.version",
        "nucleotide",
        Path("temp/data/external/sequences/nucleotides"),
    )
    download_sequences_bulk(
        non_amr,
        "protein_accession.version",
        "protein",
        Path("temp/data/external/sequences/proteins"),
    )

    # Extract fine-tuning subset
    fine_tuning_subset = non_amr.limit(fine_tuning_count)

    # Extract pre-training which is everything except the fine-tuning subset which is an anti-join on the fine-tuning subset
    pre_training_subset = non_amr.join(
        fine_tuning_subset.select("GeneID"), on="GeneID", how="anti"
    )

    # Collect both subsets to Parquet for downstream training pipelines
    pre_training_subset.sink_parquet(
        Path("temp/data/processed/pretraining/bacteria.gene.pre_training.parquet")
    )

    fine_tuning_subset.sink_parquet(
        Path("temp/data/interim/bacteria.gene.fine_tuning.parquet")
    )

    # Load each one to check size
    print(
        f"Pre-training subset size: {pl.read_parquet(Path('temp/data/processed/pretraining/bacteria.gene.pre_training.parquet')).height}"
    )
    print(
        f"Fine-tuning subset size: {pl.read_parquet(Path('temp/data/interim/bacteria.gene.fine_tuning.parquet')).height}"
    )


if __name__ == "__main__":
    main()
