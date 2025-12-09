from typing import Optional

from tarp.cli.logging import Console
from tarp.model.backbone import Encoder
from tarp.services.tokenizers import Tokenizer
from tarp.services.training.pipelines.stage import Stage


class Pipeline:
    def __init__(self, encoder: Encoder, tokenizer: Tokenizer, run_id: str) -> None:
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.run_id = run_id
        self.upcoming_stages: list[Stage] = []
        self.last_completed_stage: Optional[Stage] = None

    def __rshift__(self, other: Stage) -> "Pipeline":
        return self.pipe(other)

    def pipe(self, stage: Stage) -> "Pipeline":
        self.upcoming_stages.append(stage)
        return self

    def run(self) -> Encoder:
        current_encoder = self.encoder
        # Run upcoming stages in sequence and store completed stages removing them from upcoming
        # stages this way we can keep track of what has been done
        while self.upcoming_stages:
            stage = self.upcoming_stages.pop(0)
            Console.info(f"Starting stage: {stage.name}")
            current_encoder = stage.run(current_encoder, self.tokenizer)
            self.last_completed_stage = stage
            Console.info(f"Completed stage: {stage.name}")
        self.encoder = current_encoder
        return current_encoder
