import math
from pathlib import Path

import torch
from torch import nn
from torchmetrics.text import Perplexity

from tarp.cli.core import Console
from tarp.data.datasets.distillation.core import CrossDistillationDataset
from tarp.data.datasets.language.diffusion import CosineDiffusionMaskingDataset
from tarp.data.datasets.language.masked import PoissonSpanMaskingDataset
from tarp.data.sources.sequence import GenomeSliceSource
from tarp.evaluation.text.masked import MaskedLanguageAccuracy
from tarp.model.backbone.pretrained.esm1b import Esm1bEncoder
from tarp.model.backbone.untrained.transformer import TransformerEncoder
from tarp.model.layers.positional.core import (
    HeterogeneousTransformativePositionalEncoding,
)
from tarp.model.layers.positional.rotational import (
    CachedIntegerRotaryPositionalEncoding,
)
from tarp.model.tasks.distillation import CrossLanguageDistillationModel
from tarp.model.tasks.language import LanguageModel
from tarp.preprocessing.tokenizers.atomic.monomer import NucleotideTokenizer
from tarp.preprocessing.tokenizers.pretrained.esm1b import Esm1bTokenizer
from tarp.training.trainer.distillation.transfer import CrossLanguageDistillationTrainer
from tarp.training.trainer.language.diffusion import DiffusionLanguageModelTrainer
from tarp.training.trainer.language.masked import MaskedLanguageModelTrainer

if __name__ == "__main__":
    dna_tokenizer = NucleotideTokenizer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_masked_language_dataset = CosineDiffusionMaskingDataset(
        source=GenomeSliceSource(
            genomes_directory=Path("temp/data/external/sequences/nucleotides"),
            metadata_source=Path(
                "temp/data/processed/finetuning/fold_0/train/bacteria.gene.fine_tuning.parquet"
            ),
            key_column="genomic_nucleotide_accession.version",
            start_column="start_position_on_the_genomic_accession",
            end_column="end_position_on_the_genomic_accession",
            sequence_column="dna_sequence",
        ),
        tokenizer=dna_tokenizer,
        sequence_column="dna_sequence",
        maximum_sequence_length=768,
        masking_probability_maximum=0.15,
        masking_probability_minimum=0.05,
    )

    val_masked_language_dataset = CosineDiffusionMaskingDataset(
        source=GenomeSliceSource(
            genomes_directory=Path("temp/data/external/sequences/nucleotides"),
            metadata_source=Path(
                "temp/data/processed/finetuning/fold_0/test/bacteria.gene.fine_tuning.parquet"
            ),
            key_column="genomic_nucleotide_accession.version",
            start_column="start_position_on_the_genomic_accession",
            end_column="end_position_on_the_genomic_accession",
            sequence_column="dna_sequence",
        ),
        tokenizer=dna_tokenizer,
        sequence_column="dna_sequence",
        maximum_sequence_length=768,
        masking_probability_maximum=0.15,
        masking_probability_minimum=0.05,
    )

    embedding_dimension = 192
    encoder = TransformerEncoder(
        model_dimension=embedding_dimension,
        number_of_layers=embedding_dimension // 64,
        number_of_heads=embedding_dimension // 64,
        feed_forward_dimension=(embedding_dimension * 8) // 3,
        positional_encoder=HeterogeneousTransformativePositionalEncoding(
            CachedIntegerRotaryPositionalEncoding(64, base=10_000),
            CachedIntegerRotaryPositionalEncoding(64, base=10_000),
        ),
        dropout=0.1,
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

    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    batch_size = 16
    accumulation_steps = 16  # To achieve an effective batch size of 256
    epochs = 8
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
