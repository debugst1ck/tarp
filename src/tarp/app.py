import gc
import math
import operator
from functools import reduce
from pathlib import Path
from typing import cast

import polars as pl
import torch
import torch.multiprocessing as mp
from torch import nn, optim

from tarp.cli.core import Console
from tarp.data.datasets.classification.multilabel import MultiLabelClassificationDataset
from tarp.data.datasets.language.masked import CosineDiffusionMaskingDataset
from tarp.data.sources.sequence import (
    GenomeSliceSource,
    SequenceDataSource,
    TabularSequenceSource,
)
from tarp.model.backbone.untrained.transformer import TransformerEncoder
from tarp.model.criterion.hybrid import LabelDistributionAwareMarginLoss
from tarp.model.layers.positional.core import (
    HeterogeneousTransformativePositionalEncoding,
)
from tarp.model.layers.positional.rotational import (
    ContinuousRotaryPositionalEncoding,
)
from tarp.model.tasks.classification import ClassificationModel
from tarp.model.tasks.language import LanguageModel
from tarp.preprocessing.tokenizers.atomic.monomer import NucleotideTokenizer
from tarp.training.trainer.classification.multilabel import (
    MultiLabelClassificationTrainer,
)
from tarp.training.trainer.language.masked import MaskedLanguageModelTrainer


def main():
    Console.info("App initialized")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        Console.warning(
            "Multiprocessing start method already set, proceeding with existing method."
        )

    tokenizer = NucleotideTokenizer()

    Console.info(f"Tokenizer vocabulary size: {tokenizer.vocabulary_size}")

    # train_sources: list[SequenceDataSource[dict[str, str]]] = [
    #     GenomeSliceSource(
    #         genomes_directory=Path("temp/data/external/sequences/nucleotides"),
    #         metadata_source=Path(
    #             "temp/data/processed/pretraining/bacteria_gene_pre_training.parquet"
    #         ),
    #         key_column="genomic_nucleotide_accession.version",
    #         start_column="start_position_on_the_genomic_accession",
    #         end_column="end_position_on_the_genomic_accession",
    #         sequence_column="dna_sequence",
    #     ),
    # ]

    train_sources: list[SequenceDataSource[dict[str, str]]] = [
        TabularSequenceSource(
            source=Path("temp/data/processed/finetuning/fold_0/train/card_amr.parquet"),
        ),
        GenomeSliceSource(
            genomes_directory=Path("temp/data/external/sequences/nucleotides"),
            metadata_source=Path(
                "temp/data/processed/finetuning/fold_0/train/bacteria.gene.fine_tuning.parquet"
            ),
            key_column="genomic_nucleotide_accession.version",
            start_column="start_position_on_the_genomic_accession",
            end_column="end_position_on_the_genomic_accession",
            sequence_column="dna_sequence",
        ),
    ]
    test_sources: list[SequenceDataSource[dict[str, str]]] = [
        TabularSequenceSource(
            source=Path("temp/data/processed/finetuning/fold_0/test/card_amr.parquet"),
        ),
        GenomeSliceSource(
            genomes_directory=Path("temp/data/external/sequences/nucleotides"),
            metadata_source=Path(
                "temp/data/processed/finetuning/fold_0/test/bacteria.gene.fine_tuning.parquet"
            ),
            key_column="genomic_nucleotide_accession.version",
            start_column="start_position_on_the_genomic_accession",
            end_column="end_position_on_the_genomic_accession",
            sequence_column="dna_sequence",
        ),
    ]
    train_dataset = CosineDiffusionMaskingDataset(
        source=reduce(operator.add, train_sources),
        tokenizer=tokenizer,
        sequence_column="dna_sequence",
        maximum_sequence_length=512,
    )
    test_dataset = CosineDiffusionMaskingDataset(
        source=reduce(operator.add, test_sources),
        tokenizer=tokenizer,
        sequence_column="dna_sequence",
        maximum_sequence_length=512,
    )

    Console.debug(f"Training dataset size: {len(train_dataset):,}")
    Console.debug(f"Validation dataset size: {len(test_dataset):,}")

    encoder = TransformerEncoder(
        model_dimension=384,
        number_of_heads=6,
        number_of_layers=6,
        feed_forward_dimension=1024,
        positional_encoder=HeterogeneousTransformativePositionalEncoding(
            query_encoder=ContinuousRotaryPositionalEncoding(dimension=64, base=10_000),
            key_encoder=ContinuousRotaryPositionalEncoding(dimension=64, base=10_000),
        ),
        dropout=0.1,
        bias=False,
    )

    Console.debug(
        f"Number of parameters in the encoder: {sum(p.numel() for p in encoder.parameters()):,}"
    )

    embedding = nn.Embedding(
        num_embeddings=tokenizer.vocabulary_size,
        embedding_dim=384,
        padding_idx=tokenizer.pad_token_id,
    )

    nn.init.normal_(embedding.weight, mean=0.0, std=1 / math.sqrt(384))

    model = LanguageModel(
        embedding=embedding,
        encoder=encoder,
        vocabulary_size=tokenizer.vocabulary_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    batch_size = 16
    accumulation_steps = 16  # To achieve an effective batch size of 256
    epochs = 5
    learning_rate = 5e-4

    steps_per_epoch = math.ceil((len(train_dataset) / batch_size) / accumulation_steps)
    total_steps = steps_per_epoch * epochs
    warmup_steps = total_steps // 10  # Warmup for the first 10% of training

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

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

    trainer = MaskedLanguageModelTrainer(
        model=model,
        training_dataset=train_dataset,
        validation_dataset=test_dataset,
        optimizer=optimizer,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size=batch_size,
        criterion=nn.CrossEntropyLoss(ignore_index=-100),
        worker_count=4,
        accumulation_steps=accumulation_steps,
        persistent_workers=False,
        scheduler=scheduler,
        epochs=epochs,
    ).fit()

    batch_size = 16
    accumulation_steps = 16  # To achieve an effective batch size of 256
    epochs = 10
    learning_rate = 5e-4

    label_columns = (
        pl.read_csv(Path("temp/data/cache/labels.csv")).to_series().to_list()
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    classification_model = ClassificationModel(
        embedding=trainer.context.model.embedding,
        encoder=trainer.context.model.encoder,
        number_of_classes=len(label_columns),
    ).to(device)

    classification_training_dataset = MultiLabelClassificationDataset(
        source=cast(
            SequenceDataSource[dict[str, str | float]],
            TabularSequenceSource(
                source=Path(
                    "temp/data/processed/finetuning/fold_0/train/card_amr.parquet"
                ),
            ),
        ),
        tokenizer=tokenizer,
        sequence_column="dna_sequence",
        label_columns=label_columns,
        maximum_sequence_length=512,
    )

    classification_test_dataset = MultiLabelClassificationDataset(
        source=cast(
            SequenceDataSource[dict[str, str | float]],
            TabularSequenceSource(
                source=Path(
                    "temp/data/processed/finetuning/fold_0/test/card_amr.parquet"
                ),
            ),
        ),
        tokenizer=tokenizer,
        sequence_column="dna_sequence",
        label_columns=label_columns,
        maximum_sequence_length=512,
    )

    classification_optimizer = optim.AdamW(
        classification_model.parameters(), lr=learning_rate, weight_decay=0.01
    )

    classification_scheduler = torch.optim.lr_scheduler.SequentialLR(
        classification_optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                classification_optimizer,
                start_factor=0.01,
                total_iters=warmup_steps,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                classification_optimizer,
                T_max=total_steps - warmup_steps,
            ),
        ],
        milestones=[warmup_steps],
    )

    classification_trainer = MultiLabelClassificationTrainer(
        model=classification_model,
        training_dataset=classification_training_dataset,
        validation_dataset=classification_test_dataset,
        optimizer=classification_optimizer,
        device=device,
        batch_size=batch_size,
        criterion=criterion,
        worker_count=4,
        persistent_workers=False,
        scheduler=classification_scheduler,
        epochs=epochs,
    ).fit()

    Console.warning(f"Freeing memory, {gc.collect()} objects collected.")
    Console.info("Training completed.")


if __name__ == "__main__":
    main()
