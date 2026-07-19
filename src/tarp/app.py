import math
from pathlib import Path
from typing import cast

import torch
from torch import nn

from tarp.cli.core import Console
from tarp.data.datasets.language.diffusion import CosineDiffusionMaskingDataset
from tarp.data.sources.sequence import GenomeSliceSource
from tarp.model.backbone.untrained.transformer import TransformerEncoder
from tarp.model.layers.positional.rotational import (
    ContinuousRotaryPositionalEncoding,
)
from tarp.model.tasks.language import LanguageModel
from tarp.preprocessing.tokenizers.atomic.monomer import NucleotideTokenizer
from tarp.training.engine.single import SingleDeviceEngine
from tarp.training.objectives.language.masked import MaskedLanguageModelingObjective
from tarp.training.orchestrator.core import Orchestrator
from tarp.training.plugins.core import State
from tarp.training.plugins.scheduling import BatchLearningScheduling


def main():
    dna_tokenizer = NucleotideTokenizer()

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

    embedding_dimension = 384
    encoder = TransformerEncoder(
        model_dimension=embedding_dimension,
        number_of_layers=embedding_dimension // 32,
        number_of_heads=embedding_dimension // 64,
        feed_forward_dimension=(embedding_dimension * 8) // 3,
        positional_encoder=ContinuousRotaryPositionalEncoding(
            dimension=64, base=100_000
        ),
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
    accumulation_steps = 32
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

    orchestrator = Orchestrator(
        engine=SingleDeviceEngine(
            model=language_model,
            device_idx=0,
            mixed_precision=True,
            mixed_precision_dtype=torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16,
        ),
        objective=MaskedLanguageModelingObjective(
            criterion=criterion,
        ),
        optimizer=optimizer,
        plugins=[
            BatchLearningScheduling(
                scheduler=scheduler,
            )
        ],
        accumulation_steps=accumulation_steps,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_masked_language_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_masked_language_dataset.collate,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_masked_language_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=val_masked_language_dataset.collate,
    )

    state = State()

    for epoch in range(epochs):
        Console.info(f"Epoch {epoch + 1}/{epochs}")
        state = orchestrator.run(
            dataloader=train_dataloader, state=state, is_training=True
        )
        Console.info(f"Validation after epoch {epoch + 1}/{epochs}")
        state = orchestrator.run(
            dataloader=val_dataloader, state=state, is_training=False
        )


if __name__ == "__main__":
    main()
