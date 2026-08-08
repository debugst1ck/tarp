import math
from pathlib import Path

import torch
from torch import distributed as dist
from torch import nn

from odyssey import DistributedDataParallelRuntime, Orchestrator, State
from tarp.cli.core import Console
from tarp.data.datasets.language.masked import PoissonSpanMaskingDataset
from tarp.data.sources.sequence import GenomeSliceSource
from tarp.model.backbone.untrained.transformer import TransformerEncoder
from tarp.model.layers.positional.rotational import CachedRotaryPositionalEncoding
from tarp.model.tasks.language import LanguageModel
from tarp.preprocessing.tokenizers.atomic.monomer import NucleotideTokenizer
from tarp.training.objectives.language.masked import MaskedLanguageModelingObjective
from tarp.training.plugins.checkpointing import CheckpointOnEnd
from tarp.training.plugins.scheduling import BatchLearningScheduling
from tarp.training.plugins.tui import ProgressBar


def main():
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")

    world_size = dist.get_world_size() if dist.is_initialized() else 1

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
        maximum_sequence_length=1536,
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
        maximum_sequence_length=1536,
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

    language_model = LanguageModel(
        encoder=encoder,
        embedding=dna_embedding,
        vocabulary_size=dna_tokenizer.vocabulary_size,
    )

    engine = DistributedDataParallelRuntime(
        model=language_model,
        mixed_precision=True,
        mixed_precision_dtype=torch.bfloat16
        if torch.accelerator.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    batch_size = 16
    accumulation_steps = 48
    epochs = 5
    learning_rate_adam = 3e-4
    learning_rate_muon = 0.02

    train_sampler = torch.utils.data.DistributedSampler(
        train_masked_language_dataset,
        num_replicas=world_size,
        rank=dist.get_rank(),
        shuffle=True,
        drop_last=True,
    )

    val_sampler = torch.utils.data.DistributedSampler(
        val_masked_language_dataset,
        num_replicas=world_size,
        rank=dist.get_rank(),
        shuffle=False,
        drop_last=True,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_masked_language_dataset,
        batch_size=batch_size,
        num_workers=6,
        sampler=train_sampler,
        pin_memory=True,
        collate_fn=train_masked_language_dataset.collate,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_masked_language_dataset,
        batch_size=batch_size,
        num_workers=6,
        sampler=val_sampler,
        pin_memory=True,
        collate_fn=val_masked_language_dataset.collate,
    )

    batches_per_epoch = len(train_dataloader)
    steps_per_epoch = math.ceil(batches_per_epoch / accumulation_steps)

    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, total_steps // 10)

    muon_parameters: list[nn.Parameter] = []
    adam_parameters: list[nn.Parameter] = []

    for name, param in engine.model.named_parameters():
        if not param.requires_grad:
            continue

        if "encoder" in name and param.ndim == 2:
            muon_parameters.append(param)
        else:
            adam_parameters.append(param)

    adam = torch.optim.AdamW(
        params=adam_parameters,
        lr=learning_rate_adam,
    )

    muon = torch.optim.Muon(
        params=muon_parameters,
        lr=learning_rate_muon,
    )

    adam_scheduler = torch.optim.lr_scheduler.SequentialLR(
        adam,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                adam, start_factor=0.01, total_iters=warmup_steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                adam, T_max=total_steps - warmup_steps, eta_min=learning_rate_adam * 0.1
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
        runtime=engine,
        objective=MaskedLanguageModelingObjective(
            criterion=criterion,
        ),
        optimizers=(adam, muon),
        plugins=[
            BatchLearningScheduling(schedulers=(adam_scheduler, muon_scheduler)),
            CheckpointOnEnd(path=Path("temp/checkpoints/checkpoint_lm.safetensors")),
            ProgressBar(),
        ],
        accumulation_steps=accumulation_steps,
    )

    if engine.is_main_process:
        Console.debug(
            f"Trainable parameters: {sum(p.numel() for p in engine.model.parameters() if p.requires_grad):,}"
        )

    state = State()

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        if engine.is_main_process:
            Console.info(f"Training Epoch {epoch + 1}/{epochs}")
        state = orchestrator.run(
            dataloader=train_dataloader, state=state, is_training=True
        )
        if engine.is_main_process:
            Console.info(f"Validation after epoch {epoch + 1}/{epochs}")
        state = orchestrator.run(
            dataloader=val_dataloader, state=state, is_training=False
        )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
