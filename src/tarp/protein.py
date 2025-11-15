import datetime
import os
import torch
from pathlib import Path

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import polars as pl

from tarp.model.backbone import Encoder
from tarp.model.backbone.untrained.hyena import HyenaEncoder
from tarp.model.backbone.untrained.transformer import TransformerEncoder

from tarp.model.finetuning.language import LanguageModel
from tarp.model.finetuning.metric.triplet import TripletMetricModel
from tarp.model.finetuning.classification import ClassificationModel

from tarp.services.utilities.seed import establish_random_seed
from tarp.cli.logging import Console
from tarp.services.tokenizers.pretrained.character import CharacterTokenizer
from tarp.services.datasource.sequence import (
    TabularSequenceSource,
    FastaSliceSource,
)
from tarp.services.datasets.classification.multilabel import (
    MultiLabelClassificationDataset,
)
from tarp.services.datasets.metric.triplet import MultiLabelOfflineTripletDataset
from tarp.services.training.trainer.classification.multilabel import (
    MultiLabelClassificationTrainer,
)
from tarp.services.evaluation.losses.multilabel import AsymmetricFocalLoss
from tarp.services.training.trainer.metric.triplet import OfflineTripletMetricTrainer
from tarp.services.preprocessing.augmentation.nucleotide import (
    RandomMutation,
    InsertionDeletion,
    ReverseComplement,
)
from tarp.services.preprocessing.augmentation import CompositeAugmentation

from tarp.services.datasets.language.masked import MaskedLanguageModelDataset

from tarp.config import TransformerConfig

from tarp.services.training.trainer.language.masked import MaskedLanguageModelTrainer


import torch.multiprocessing as mp


def pretrain(device: torch.device, encoder: Encoder) -> Encoder:
    masked_language_dataset_train = MaskedLanguageModelDataset(
        data_source=(
            FastaSliceSource(
                directory=Path("temp/data/external/sequences/proteins"),
                metadata=Path("temp/data/processed/pre_training.parquet"),
                key_column="protein_accession.version",
                start_column="?",
                end_column="?",
                orientation_column="orientation",
                sequence_column="protein_sequence",
            )
        ),
        tokenizer=CharacterTokenizer(),
        sequence_column="protein_sequence",
        maximum_sequence_length=512,
        masking_probability=0.15,
        augmentation=CompositeAugmentation(
            [
                RandomMutation(),
                InsertionDeletion(),
                ReverseComplement(0.5),
            ]
        ),
    )

    masked_language_dataset_valid = MaskedLanguageModelDataset(
        data_source=(
            FastaSliceSource(
                directory=Path("temp/data/external/sequences/proteins"),
                metadata=Path("temp/data/processed/fine_tuning.valid.parquet"),
                key_column="protein_accession.version",
                start_column="?",
                end_column="?",
                orientation_column="orientation",
            )
        ),
        tokenizer=CharacterTokenizer(),
        sequence_column="protein_sequence",
        maximum_sequence_length=512,
        masking_probability=0.15,
    )

    language_model = LanguageModel(
        encoder=encoder,
        vocabulary_size=masked_language_dataset_train.tokenizer.vocab_size,
    )

    torch.compile(language_model, mode="reduce-overhead")

    optimizer_language = AdamW(
        [
            {
                "params": language_model.encoder.parameters(),
                "lr": 1e-4,
                "weight_decay": 1e-2,
            },
            {
                "params": language_model.language_head.parameters(),
                "lr": 1e-3,
                "weight_decay": 1e-2,
            },
        ]
    )

    Console.info("Starting masked language model training")
    Console.debug(
        f"Vocabulary size: {masked_language_dataset_train.tokenizer.vocab_size}"
    )
    MaskedLanguageModelTrainer(
        model=language_model,
        train_dataset=masked_language_dataset_train,
        valid_dataset=masked_language_dataset_valid,
        optimizer=optimizer_language,
        scheduler=CosineAnnealingWarmRestarts(optimizer_language, T_0=5, T_mult=2),
        device=device,
        epochs=20,
        num_workers=4,
        batch_size=64,
        accumulation_steps=4,
        persistent_workers=(
            True if mp.get_start_method(allow_none=True) == "spawn" else False
        ),
        vocabulary_size=masked_language_dataset_train.tokenizer.vocab_size,
    ).fit()

    return language_model.encoder


def finetune(device: torch.device, encoder: Encoder) -> Encoder:
    label_columns = (
        pl.read_csv(Path("temp/data/cache/labels.csv")).to_series().to_list()
    )

    multilabel_classification_train = MultiLabelClassificationDataset(
        (
            TabularSequenceSource(
                source=Path("temp/data/processed/card_amr.train.parquet"),
            )
            + FastaSliceSource(
                directory=Path("temp/data/external/sequences/proteins"),
                metadata=Path("temp/data/processed/fine_tuning.train.parquet"),
                key_column="protein_accession.version",
                start_column="?",
                end_column="?",
                orientation_column="orientation",
                sequence_column="protein_sequence",
            )
        ),
        CharacterTokenizer(),
        sequence_column="protein_sequence",
        label_columns=label_columns,
        maximum_sequence_length=512,
        augmentation=CompositeAugmentation(
            [
                RandomMutation(),
                InsertionDeletion(),
                ReverseComplement(0.5),
            ]
        ),
    )

    multilabel_classification_valid = MultiLabelClassificationDataset(
        (
            TabularSequenceSource(
                source=Path("temp/data/processed/card_amr.valid.parquet")
            )
            + FastaSliceSource(
                directory=Path("temp/data/external/sequences/proteins"),
                metadata=Path("temp/data/processed/fine_tuning.valid.parquet"),
                key_column="protein_accession.version",
                start_column="?",
                end_column="?",
                orientation_column="orientation",
                sequence_column="protein_sequence",
            )
        ),
        CharacterTokenizer(),
        sequence_column="protein_sequence",
        label_columns=label_columns,
        maximum_sequence_length=512,
    )

    triplet_dataset_train = MultiLabelOfflineTripletDataset(
        base_dataset=multilabel_classification_train,
        label_cache=Path("temp/data/cache/protein_labels_cache_train.parquet"),
    )

    triplet_dataset_valid = MultiLabelOfflineTripletDataset(
        base_dataset=multilabel_classification_valid,
        label_cache=Path("temp/data/cache/protein_labels_cache_valid.parquet"),
    )

    classification_model = ClassificationModel(
        encoder=encoder,
        number_of_classes=len(label_columns),
    )

    triplet_model = TripletMetricModel(
        encoder=encoder,
    )

    torch.compile(classification_model, mode="reduce-overhead")
    torch.compile(triplet_model, mode="reduce-overhead")

    multi_triplet_optimizer = AdamW(
        triplet_model.parameters(), lr=2e-5, weight_decay=1e-3
    )
    OfflineTripletMetricTrainer(
        model=triplet_model,
        train_dataset=triplet_dataset_train,
        valid_dataset=triplet_dataset_valid,
        optimizer=multi_triplet_optimizer,
        scheduler=CosineAnnealingWarmRestarts(multi_triplet_optimizer, T_0=3, T_mult=2),
        device=device,
        epochs=6,
        num_workers=4,
        batch_size=64,
        accumulation_steps=4,
        persistent_workers=(
            True if mp.get_start_method(allow_none=True) == "spawn" else False
        ),
    ).fit()
    

    optimizer_classification = AdamW(
        [
            {
                "params": classification_model.encoder.parameters(),
                "lr": 1e-5,
                "weight_decay": 1e-2,
            },
            {
                "params": classification_model.classification_head.parameters(),
                "lr": 1e-3,
                "weight_decay": 1e-2,
            },
        ]
    )
    
    label_cache = pl.read_parquet(Path("temp/data/cache/protein_labels_cache_train.parquet"))
    label_tensor = torch.tensor(label_cache.select(label_columns).to_numpy())

    class_counts = label_tensor.sum(dim=0)
    total_counts = label_tensor.size(0)
    class_weights = (total_counts - class_counts) / class_counts
    class_weights = (class_weights - class_weights.min()) / (
        class_weights.max() - class_weights.min()
    )
    class_weights = class_weights * (10.0 - 0.1) + 0.1  # Scale to [0.1, 10.0]
    
    Console.debug(
        str(
            pl.DataFrame(
                {
                    "label": label_columns,
                    "class_weights": class_weights.tolist(),
                }
            )
        )
    )
    
    trainer = MultiLabelClassificationTrainer(
        model=classification_model,
        train_dataset=multilabel_classification_train,
        valid_dataset=multilabel_classification_valid,
        optimizer=optimizer_classification,
        scheduler=CosineAnnealingWarmRestarts(
            optimizer_classification, T_0=5, T_mult=2
        ),
        criterion=AsymmetricFocalLoss(
            gamma_neg=1, gamma_pos=3, class_weights=class_weights
        ),
        device=device,
        epochs=15,
        num_workers=4,
        batch_size=64,
        accumulation_steps=4,
        persistent_workers=(
            True if mp.get_start_method(allow_none=True) == "spawn" else False
        ),
    )
    trainer.fit()
    
    return classification_model.encoder


def save_model(model: torch.nn.Module, name: str, run_id: str) -> Path:
    path = Path(f"temp/checkpoints/{name}_{run_id}.pt")
    # Move model to CPU before saving to avoid CUDA initialization issues on load
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)
    return path


def load_model(
    model: torch.nn.Module, name: str, run_id: str, device: torch.device
) -> torch.nn.Module:
    path = Path(f"temp/checkpoints/{name}_{run_id}.pt")
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def main() -> None:
    Console.info("App started")

    try:
        mp.set_start_method("spawn", force=True)
        Console.info("Multiprocessing start method set to 'spawn'")
    except RuntimeError:
        Console.warning("Multiprocessing start method was already set, skipping...")
        pass

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set seed for reproducibility
    SEED = establish_random_seed(69420)  # FuNnY NuMbEr :D
    Console.info(f"Random seed set to {SEED}")

    encoder = TransformerEncoder(
        vocabulary_size=CharacterTokenizer().vocab_size,
        embedding_dimension=TransformerConfig.embedding_dimension,
        hidden_dimension=TransformerConfig.hidden_dimension,
        padding_id=CharacterTokenizer().pad_token_id,
        number_of_layers=TransformerConfig.number_of_layers,
        number_of_heads=TransformerConfig.number_of_heads,
        dropout=TransformerConfig.dropout,
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
    pretrained_encoder = pretrain(device=device, encoder=encoder)
    
    Console.info("Starting fine-tuning phase")
    finetuned_encoder = finetune(device=device, encoder=pretrained_encoder)

    save_model(
        model=finetuned_encoder,
        name="classification_model_encoder",
        run_id=run_id,
    )

    Console.info("Training complete")

if __name__ == "__main__":
    main()
