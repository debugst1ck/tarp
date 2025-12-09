from abc import ABC, abstractmethod
from typing import Optional

import torch

from tarp.model.backbone import Encoder
from tarp.services.datasource.sequence import SequenceDataSource
from tarp.services.tokenizers import Tokenizer


class Stage(ABC):
    def __init__(self, name: str, run_id: str, device: torch.device) -> None:
        self.name = name
        self.run_id = run_id
        self.device = device
        self.params: dict = {}
        self.train_source: Optional[SequenceDataSource] = None
        self.valid_source: Optional[SequenceDataSource] = None
        self._model: Optional[torch.nn.Module] = None

    def with_sources(
        self, train_source: SequenceDataSource, valid_source: SequenceDataSource
    ) -> "Stage":
        self.train_source = train_source
        self.valid_source = valid_source
        return self

    def with_parameters(self, **params) -> "Stage":
        self.params = params
        return self

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            raise ValueError("Model has not been trained yet")
        return self._model

    @abstractmethod
    def run(self, encoder: Encoder, tokenizer: Tokenizer) -> Encoder:
        raise NotImplementedError
