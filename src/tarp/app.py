import math
from pathlib import Path

import polars as pl
import torch
from torch import nn
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.text import Perplexity

from tarp.cli.core import Console
from tarp.data.datasets.classification.multilabel import MultiLabelClassificationDataset
from tarp.data.datasets.language.diffusion import CosineDiffusionMaskingDataset
from tarp.data.sources.sequence import GenomeSliceSource, TabularSequenceSource
from tarp.evaluation.text.masked import MaskedLanguageAccuracy
from tarp.model.backbone.untrained.transformer import TransformerEncoder
from tarp.model.layers.positional.core import (
    HeterogeneousTransformativePositionalEncoding,
)
from tarp.model.layers.positional.rotational import (
    CachedIntegerRotaryPositionalEncoding,
)
from tarp.model.tasks.classification import ClassificationModel
from tarp.model.tasks.language import LanguageModel
from tarp.preprocessing.tokenizers.atomic.monomer import NucleotideTokenizer
from tarp.training.trainer.classification.multilabel import (
    MultiLabelClassificationTrainer,
)
from tarp.training.trainer.language.diffusion import DiffusionLanguageModelTrainer


def main():
    dna_tokenizer = NucleotideTokenizer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_masked_language_dataset = CosineDiffusionMaskingDataset(
        source=GenomeSliceSource(
            genomes_directory=Path("temp/data/external/sequences/nucleotides"),
            metadata_source=Path(
                "temp/data/processed/pretraining/bacteria.gene.pre_training.parquet"
            ),
            key_column="genomic_nucleotide_accession.version",
            start_column="start_position_on_the_genomic_accession",
            end_column="end_position_on_the_genomic_accession",
            sequence_column="dna_sequence",
        ),
        tokenizer=dna_tokenizer,
        sequence_column="dna_sequence",
        masking_probability_maximum=0.15,
        masking_probability_minimum=0.05,
        maximum_sequence_length=1024,
    )

    val_masked_language_dataset = CosineDiffusionMaskingDataset(
        source=GenomeSliceSource(
            genomes_directory=Path("temp/data/external/sequences/nucleotides"),
            metadata_source=Path(
                "temp/data/processed/finetuning/fold_0/val/bacteria.gene.fine_tuning.parquet"
            ),
            key_column="genomic_nucleotide_accession.version",
            start_column="start_position_on_the_genomic_accession",
            end_column="end_position_on_the_genomic_accession",
            sequence_column="dna_sequence",
        ),
        tokenizer=dna_tokenizer,
        sequence_column="dna_sequence",
        masking_probability_maximum=0.16,
        masking_probability_minimum=0.14,
        maximum_sequence_length=1024,
    )

    embedding_dimension = 576
    encoder = TransformerEncoder(
        model_dimension=embedding_dimension,
        number_of_layers=embedding_dimension // 32,
        number_of_heads=embedding_dimension // 64,
        feed_forward_dimension=(embedding_dimension * 8) // 3,
        positional_encoder=HeterogeneousTransformativePositionalEncoding(
            CachedIntegerRotaryPositionalEncoding(64, base=10_000),
            CachedIntegerRotaryPositionalEncoding(64, base=10_000),
        ),
        dropout=0.1,
    )

    Console.debug(
        f"Trainable parameters: {sum(p.numel() for p in encoder.parameters() if p.requires_grad):,}"
    )

    dna_embedding = nn.Embedding(
        num_embeddings=dna_tokenizer.vocabulary_size,
        embedding_dim=embedding_dimension,
        padding_idx=dna_tokenizer.pad_token_id,
    )

    _ = nn.init.normal_(
        dna_embedding.weight, mean=0.0, std=1 / math.sqrt(embedding_dimension)
    )

    language_model = LanguageModel(
        encoder=encoder,
        embedding=dna_embedding,
        vocabulary_size=dna_tokenizer.vocabulary_size,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    batch_size = 16
    accumulation_steps = 16  # To achieve an effective batch size of 256
    epochs = 5
    learning_rate = 5e-4

    steps_per_epoch = math.ceil(
        len(train_masked_language_dataset) / (batch_size * accumulation_steps)
    )
    total_steps = steps_per_epoch * epochs
    warmup_steps = total_steps // 10  # Warmup for the first 10% of training

    optimizer = torch.optim.AdamW(
        params=language_model.parameters(), lr=learning_rate, weight_decay=1e-2
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps - warmup_steps
            ),
        ],
        milestones=[warmup_steps],
    )

    trainer = DiffusionLanguageModelTrainer(
        model=language_model,
        training_dataset=train_masked_language_dataset,
        validation_dataset=val_masked_language_dataset,
        optimizer=optimizer,
        device=device,
        criterion=criterion,
        scheduler=scheduler,
        batch_size=batch_size,
        epochs=epochs,
        worker_count=4,
        accumulation_steps=accumulation_steps,
        metrics=(
            Perplexity(ignore_index=-100),
            MaskedLanguageAccuracy(
                ignore_index=-100,
                num_classes=dna_tokenizer.vocabulary_size,
            ),
        ),
    ).fit()

    label_columns = (
        pl.read_csv(Path("temp/data/cache/labels_reduced.csv")).to_series().to_list()
    )
    label_columns.remove("non_amr")

    train_multilabel_dataset = MultiLabelClassificationDataset(
        source=TabularSequenceSource(
            Path("temp/data/processed/finetuning/fold_0/train/card_amr.parquet")
        ),
        tokenizer=dna_tokenizer,
        sequence_column="dna_sequence",
        label_columns=label_columns,
        maximum_sequence_length=1024,
    )

    val_multilabel_dataset = MultiLabelClassificationDataset(
        source=TabularSequenceSource(
            Path("temp/data/processed/finetuning/fold_0/val/card_amr.parquet")
        ),
        tokenizer=dna_tokenizer,
        sequence_column="dna_sequence",
        label_columns=label_columns,
        maximum_sequence_length=1024,
    )

    multilabel_model = ClassificationModel(
        encoder=language_model.encoder.freeze(),
        embedding=language_model.embedding,
        number_of_classes=len(label_columns),
    )

    batch_size = 16
    accumulation_steps = 16  # To achieve an effective batch size of 256
    epochs = 10  # Fine-tuning for 10 epochs
    learning_rate = 5e-4

    steps_per_epoch = math.ceil(
        len(train_masked_language_dataset) / (batch_size * accumulation_steps)
    )
    total_steps = steps_per_epoch * epochs
    warmup_steps = total_steps // 10  # Warmup for the first 10% of training

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        params=multilabel_model.parameters(), lr=learning_rate, weight_decay=1e-2
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps - warmup_steps
            ),
        ],
        milestones=[warmup_steps],
    )

    trainer = MultiLabelClassificationTrainer(
        model=multilabel_model,
        training_dataset=train_multilabel_dataset,
        validation_dataset=val_multilabel_dataset,
        optimizer=optimizer,
        device=device,
        criterion=criterion,
        scheduler=scheduler,
        batch_size=batch_size,
        epochs=epochs,
        worker_count=4,
        accumulation_steps=accumulation_steps,
        metrics=(
            Accuracy(task="multilabel", num_labels=len(label_columns)),
            Precision(task="multilabel", num_labels=len(label_columns)),
            Recall(task="multilabel", num_labels=len(label_columns)),
            F1Score(task="multilabel", num_labels=len(label_columns)),
        ),
    ).fit()

    # Save the trained model
    model_save_dir = Path("temp/models/")

    model_save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(multilabel_model.state_dict(), model_save_dir / "multilabel_model.pth")

    torch.save(language_model.state_dict(), model_save_dir / "language_model.pth")


if __name__ == "__main__":
    main()
