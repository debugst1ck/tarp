from typing import final

from tarp.data.datasets.classification.core import ClassificationDataset


@final
class MultiLabelClassificationDataset(ClassificationDataset):
    """
    A dataset for multi-label classification tasks.
    """
