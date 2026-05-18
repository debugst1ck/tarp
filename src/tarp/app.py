import operator
from functools import reduce
from pathlib import Path

import torch
from torch import nn, optim

from tarp.cli.core import Console
from tarp.data.datasets.language.masked import MaskedLanguageDataset
from tarp.data.sources.sequence import (
    GenomeSliceSource,
    SequenceDataSource,
    TabularSequenceSource,
)
from tarp.model.backbone.untrained.compression import (
    ElasticCompressedTransformerEncoder,
)
from tarp.model.backbone.untrained.transformer import TransformerEncoder
from tarp.model.heads.language import LanguageModel
from tarp.model.layers.positional.core import (
    HeterogeneousTransformativePositionalEncoding,
)
from tarp.model.layers.positional.rotational import (
    CachedIntegerRotaryPositionalEncoding,
    ContinuousRotaryPositionalEncoding,
)
from tarp.preprocessing.tokenizers.fixed.monomer import NucleotideTokenizer
from tarp.training.trainer.language.masked import MaskedLanguageModelTrainer


def main():
    Console.info("App initialized")

    tokenizer = NucleotideTokenizer()

    Console.info(f"Tokenizer vocabulary size: {tokenizer.vocabulary_size}")

    train_sources: list[SequenceDataSource[dict[str, str]]] = [
        TabularSequenceSource(
            source=Path("temp/data/processed/finetuning/fold_0/train/card_amr.parquet"),
        ),
        GenomeSliceSource(
            genomes_directory=Path("temp/data/external/sequences/nucleotides"),
            metadata_source=Path(
                "temp/data/processed/finetuning/fold_0/train/bacteria_gene_fine_tuning.parquet"
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
                "temp/data/processed/finetuning/fold_0/test/bacteria_gene_fine_tuning.parquet"
            ),
            key_column="genomic_nucleotide_accession.version",
            start_column="start_position_on_the_genomic_accession",
            end_column="end_position_on_the_genomic_accession",
            sequence_column="dna_sequence",
        ),
    ]
    train_dataset = MaskedLanguageDataset(
        source=reduce(operator.add, train_sources),
        tokenizer=tokenizer,
        sequence_column="dna_sequence",
        maximum_sequence_length=2048,
    )
    test_dataset = MaskedLanguageDataset(
        source=reduce(operator.add, test_sources),
        tokenizer=tokenizer,
        sequence_column="dna_sequence",
        maximum_sequence_length=2048,
    )

    Console.debug(f"Training dataset size: {len(train_dataset)}")
    Console.debug(f"Validation dataset size: {len(test_dataset)}")

    # encoder = TransformerEncoder(
    #     model_dimension=384,
    #     number_of_heads=6,
    #     number_of_layers=6,
    #     feed_forward_dimension=1024,  # 8/3 for swiglu
    #     dropout=0.1,
    #     positional_encoder=HeterogeneousTransformativePositionalEncoding(
    #         CachedIntegerRotaryPositionalEncoding(
    #             dimension=384 // 6,
    #             base=10_000,
    #         ),
    #         CachedIntegerRotaryPositionalEncoding(
    #             dimension=384 // 6,
    #             base=10_000,
    #         ),
    #     ),
    # )

    encoder = ElasticCompressedTransformerEncoder(
        model_dimension=384,
        number_of_heads=6,
        number_of_layers=6,
        feed_forward_dimension=1024,  # 8/3 for swiglu
        dropout=0.1,
        positional_encoder=HeterogeneousTransformativePositionalEncoding(
            ContinuousRotaryPositionalEncoding(
                dimension=384 // 6,
                base=10_000,
            ),
            ContinuousRotaryPositionalEncoding(
                dimension=384 // 6,
                base=10_000,
            ),
        ),
        resolution=0.3,
        locality_radius=6,
        positional_weight=0.2,
        background_cost_payload=2.0,
        minimum_budget_usage=0.5,
    )

    Console.debug(
        f"Number of parameters in the encoder: {sum(p.numel() for p in encoder.parameters())}"
    )

    embedding = nn.Embedding(
        num_embeddings=tokenizer.vocabulary_size, embedding_dim=384
    )
    model = LanguageModel(
        embedding=embedding,
        encoder=encoder,
        vocabulary_size=tokenizer.vocabulary_size,
    )

    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_kwargs = {
        "attention_mask": torch.ones(16, 1024, dtype=torch.bool, device=device),
        "payload_mask": torch.ones(16, 1024, dtype=torch.bool, device=device),
    }
    trainer = MaskedLanguageModelTrainer(
        model=model,
        training_dataset=train_dataset,
        validation_dataset=test_dataset,
        optimizer=optim.Adafactor(model.parameters(), lr=1e-4, weight_decay=1e-2),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size=16,
        criterion=nn.CrossEntropyLoss(ignore_index=-100),
        worker_count=4,
        persistent_workers=True,
    ).fit()


if __name__ == "__main__":
    main()
