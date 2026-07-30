import math
from pathlib import Path

import torch
from torch import nn

from odyssey import AcceleratedRuntime, Orchestrator, State
from tarp.cli.core import Console
from tarp.data.datasets.language.masked import PoissonSpanMaskingDataset
from tarp.data.sources.sequence import GenomeSliceSource
from tarp.model.backbone.untrained.transformer import TransformerEncoder
from tarp.model.layers.positional.rotational import CachedRotaryPositionalEncoding
from tarp.model.tasks.language import LanguageModel
from tarp.preprocessing.tokenizers.atomic.monomer import NucleotideTokenizer
from tarp.training.objectives.language.masked import MaskedLanguageModelingObjective
from tarp.training.plugins.scheduling import BatchLearningScheduling


def main():
    dna_tokenizer = NucleotideTokenizer()

    train_masked_language_dataset = PoissonSpanMaskingDataset(
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
        maximum_sequence_length=1024,
        static_sequence_length=False,
    )

    val_masked_language_dataset = PoissonSpanMaskingDataset(
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
        maximum_sequence_length=1024,
        static_sequence_length=False,
    )

    embedding_dimension = 384
    encoder = TransformerEncoder(
        model_dimension=embedding_dimension,
        number_of_layers=embedding_dimension // 32,
        number_of_heads=embedding_dimension // 64,
        feed_forward_dimension=(embedding_dimension * 8) // 3,
        positional_encoder=CachedRotaryPositionalEncoding(dimension=64, base=10_000),
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

    engine = AcceleratedRuntime(
        model=language_model,
        mixed_precision=True,
        mixed_precision_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16,
    )

    if engine.is_main_process == 0:
        Console.debug(
            f"Trainable parameters: {sum(p.numel() for p in encoder.parameters() if p.requires_grad):,}"
        )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    batch_size = 32
    accumulation_steps = 48
    epochs = 5
    learning_rate_adam = 2e-4
    learning_rate_muon = 0.02

    steps_per_epoch = math.ceil(
        len(train_masked_language_dataset) / (batch_size * accumulation_steps)
    )
    total_steps = steps_per_epoch * epochs
    warmup_steps = total_steps // 10  # Warmup for the first 10% of training

    muon_parameters = {
        p for p in engine.model.encoder.parameters() if p.ndim == 2 and p.requires_grad
    }

    adamw_parameters = {
        p for p in engine.model.encoder.parameters() if p.ndim != 2 and p.requires_grad
    }

    adamw_parameters.update(
        p
        for p in engine.model.parameters()
        if p not in muon_parameters and p.requires_grad
    )

    adamw = torch.optim.AdamW(
        params=adamw_parameters,
        lr=learning_rate_adam,
    )

    muon = torch.optim.Muon(
        params=muon_parameters,
        lr=learning_rate_muon,
    )

    adam_scheduler = torch.optim.lr_scheduler.SequentialLR(
        adamw,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                adamw, start_factor=0.01, total_iters=warmup_steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                adamw,
                T_max=total_steps - warmup_steps,
                eta_min=learning_rate_adam * 0.1,
            ),
        ],
        milestones=[warmup_steps],
    )

    muon_scheduler = torch.optim.lr_scheduler.SequentialLR(
        muon,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                muon, start_factor=0.01, total_iters=warmup_steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                muon, T_max=total_steps - warmup_steps, eta_min=learning_rate_muon * 0.1
            ),
        ],
        milestones=[warmup_steps],
    )

    orchestrator = Orchestrator(
        engine=engine,
        objective=MaskedLanguageModelingObjective(
            criterion=criterion,
        ),
        optimizers=(adamw, muon),
        plugins=[
            BatchLearningScheduling(
                schedulers=(adam_scheduler, muon_scheduler),
            ),
        ],
        accumulation_steps=accumulation_steps,
    )

    if engine.is_main_process:
        # Print the number of trainable parameters for the model
        Console.debug(
            f"Trainable parameters: {sum(p.numel() for p in engine.model.parameters() if p.requires_grad):,}"
        )

    train_dataloader = torch.utils.data.DataLoader(
        train_masked_language_dataset,
        batch_size=batch_size,
        num_workers=6,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_masked_language_dataset.collate,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_masked_language_dataset,
        batch_size=batch_size,
        num_workers=6,
        pin_memory=True,
        drop_last=True,
        collate_fn=val_masked_language_dataset.collate,
    )

    state = State()

    for epoch in range(epochs):
        if engine.is_main_process:
            Console.info(f"Epoch {epoch + 1}/{epochs}")
        state = orchestrator.run(
            dataloader=train_dataloader, state=state, is_training=True
        )
        if engine.is_main_process:
            Console.info(f"Validation after epoch {epoch + 1}/{epochs}")
        state = orchestrator.run(
            dataloader=val_dataloader, state=state, is_training=False
        )


if __name__ == "__main__":
    main()
