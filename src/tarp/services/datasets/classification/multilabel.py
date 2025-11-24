from tarp.services.datasets.classification import ClassificationDataset
from tarp.services.datasource.sequence import SequenceDataSource
from tarp.services.preprocessing.augmentation import Augmentation, NoAugmentation
from tarp.services.tokenizers import Tokenizer


class MultiLabelClassificationDataset(ClassificationDataset):
    def __init__(
        self,
        data_source: SequenceDataSource,
        tokenizer: Tokenizer,
        sequence_column: str,
        label_columns: list[str],
        maximum_sequence_length: int,
        augmentation: Augmentation = NoAugmentation(),
    ):
        super().__init__(
            data_source,
            tokenizer,
            sequence_column,
            label_columns,
            maximum_sequence_length,
            augmentation,
        )
