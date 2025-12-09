import datetime
from pathlib import Path

import torch
import torch.multiprocessing as mp

from tarp.cli.logging import Console
from tarp.config import TransformerConfig
from tarp.model.backbone.untrained.transformer import TransformerEncoder
from tarp.model.finetuning.classification import ClassificationModel
from tarp.services.datasource.sequence import FastaSliceSource, TabularSequenceSource
from tarp.services.tokenizers.pretrained.dnabert2 import Dnabert2Tokenizer
from tarp.services.training.pipelines import Pipeline
from tarp.services.training.pipelines.stage.finetuning.multilabel import (
    MultiLabelClassificationFinetuningStage,
)
from tarp.services.training.pipelines.stage.finetuning.triplet import (
    TripletMetricFinetuningStage,
)
from tarp.services.training.pipelines.stage.pretraining.masked import (
    MaskedLanguageModelPretrainingStage,
)
from tarp.services.utilities.seed import establish_random_seed


def save_model(model: torch.nn.Module, name: str, run_id: str) -> Path:
    path = Path(f"temp/checkpoints/{run_id}/{name}_{run_id}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Move model to CPU before saving to avoid CUDA initialization issues on load
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)
    return path


def load_model(
    model: torch.nn.Module, name: str, run_id: str, device: torch.device
) -> torch.nn.Module:
    path = Path(f"temp/checkpoints/{run_id}/{name}_{run_id}.pt")
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def main() -> None:
    Console.info("App started")

    try:
        mp.set_start_method(method="spawn", force=True)
        Console.info("Multiprocessing start method set to 'spawn'")
    except RuntimeError:
        Console.warning("Multiprocessing start method was already set, skipping...")
        pass

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set seed for reproducibility
    SEED = establish_random_seed(69420)  # FuNnY NuMbEr :D
    Console.info(f"Random seed set to {SEED}")

    tokenizer = Dnabert2Tokenizer()

    encoder = TransformerEncoder(
        vocabulary_size=tokenizer.vocab_size,
        embedding_dimension=TransformerConfig.embedding_dimension,
        feedforward_dimension=TransformerConfig.feedforward_dimension,
        padding_id=tokenizer.pad_token_id,
        number_of_layers=TransformerConfig.number_of_layers,
        number_of_heads=TransformerConfig.number_of_heads,
        dropout=TransformerConfig.dropout,
    )

    # encoder = HyenaEncoder(
    #     vocabulary_size=Dnabert2Tokenizer().vocab_size,
    #     model_dimension=HyenaConfig.model_dimension,
    #     padding_id=Dnabert2Tokenizer().pad_token_id,
    #     number_of_layers=HyenaConfig.number_of_layers,
    #     number_of_heads=HyenaConfig.number_of_heads,
    #     recurrence_depth=HyenaConfig.recurrence_depth,
    #     mixing_width=HyenaConfig.mixing_width,
    #     local_context_size=HyenaConfig.local_context_size,
    #     dropout=HyenaConfig.dropout,
    # )

    # Training
    Console.debug(f"Model architecture: {encoder.__class__.__name__}")

    # Number of parameters
    Console.debug(
        f"Number of parameters: {sum(p.numel() for p in encoder.parameters() if p.requires_grad)}"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        Console.info(
            f"Using GPU: {torch.cuda.get_device_name(0)} for blazingly fast training"
        )
        device = torch.device("cuda")
    else:
        Console.warning("Using CPU for training, this may be slow")
        device = torch.device("cpu")

    Console.info("Starting pre-training phase")
    pipeline = Pipeline(encoder, run_id=run_id, tokenizer=tokenizer)

    pipeline = pipeline >> MaskedLanguageModelPretrainingStage(
        run_id, device
    ).with_sources(
        train_source=FastaSliceSource(
            directory=Path("temp/data/external/sequences/nucleotides"),
            metadata=Path("temp/data/processed/pre_training.parquet"),
            key_column="genomic_nucleotide_accession.version",
            start_column="start_position_on_the_genomic_accession",
            end_column="end_position_on_the_genomic_accession",
            orientation_column="orientation",
            sequence_column="dna_sequence",
        ),
        valid_source=FastaSliceSource(
            directory=Path("temp/data/external/sequences/nucleotides"),
            metadata=Path("temp/data/processed/fine_tuning.valid.parquet"),
            key_column="genomic_nucleotide_accession.version",
            start_column="start_position_on_the_genomic_accession",
            end_column="end_position_on_the_genomic_accession",
            orientation_column="orientation",
            sequence_column="dna_sequence",
        ),
    )

    pretrained_encoder = pipeline.run()

    Console.info("Starting fine-tuning phase")

    pipeline = pipeline >> TripletMetricFinetuningStage(run_id, device).with_sources(
        train_source=(
            TabularSequenceSource(
                source=Path("temp/data/processed/card_amr.train.parquet")
            )
            + FastaSliceSource(
                directory=Path("temp/data/external/sequences/nucleotides"),
                metadata=Path("temp/data/processed/fine_tuning.train.parquet"),
                key_column="genomic_nucleotide_accession.version",
                start_column="start_position_on_the_genomic_accession",
                end_column="end_position_on_the_genomic_accession",
                orientation_column="orientation",
                sequence_column="dna_sequence",
            )
        ),
        valid_source=(
            TabularSequenceSource(
                source=Path("temp/data/processed/card_amr.valid.parquet")
            )
            + FastaSliceSource(
                directory=Path("temp/data/external/sequences/nucleotides"),
                metadata=Path("temp/data/processed/fine_tuning.valid.parquet"),
                key_column="genomic_nucleotide_accession.version",
                start_column="start_position_on_the_genomic_accession",
                end_column="end_position_on_the_genomic_accession",
                orientation_column="orientation",
                sequence_column="dna_sequence",
            )
        ),
    )

    pipeline = pipeline >> MultiLabelClassificationFinetuningStage(
        run_id, device
    ).with_sources(
        train_source=(
            TabularSequenceSource(
                source=Path("temp/data/processed/card_amr.train.parquet")
            )
            + FastaSliceSource(
                directory=Path("temp/data/external/sequences/nucleotides"),
                metadata=Path("temp/data/processed/fine_tuning.train.parquet"),
                key_column="genomic_nucleotide_accession.version",
                start_column="start_position_on_the_genomic_accession",
                end_column="end_position_on_the_genomic_accession",
                orientation_column="orientation",
                sequence_column="dna_sequence",
            )
        ),
        valid_source=(
            TabularSequenceSource(
                source=Path("temp/data/processed/card_amr.valid.parquet")
            )
            + FastaSliceSource(
                directory=Path("temp/data/external/sequences/nucleotides"),
                metadata=Path("temp/data/processed/fine_tuning.valid.parquet"),
                key_column="genomic_nucleotide_accession.version",
                start_column="start_position_on_the_genomic_accession",
                end_column="end_position_on_the_genomic_accession",
                orientation_column="orientation",
                sequence_column="dna_sequence",
            )
        ),
    )

    finetuned_encoder = pipeline.run()

    save_model(
        model=pretrained_encoder,
        name="language_model_encoder",
        run_id=run_id,
    )

    save_model(
        model=finetuned_encoder,
        name="classification_model_encoder",
        run_id=run_id,
    )

    save_model(
        model=pipeline.last_completed_stage.model
        if pipeline.last_completed_stage
        else ClassificationModel(encoder=finetuned_encoder, number_of_classes=1),
        name="classification_model_full",
        run_id=run_id,
    )

    Console.info("Training complete")
    return


if __name__ == "__main__":
    main()
    exit(0)
