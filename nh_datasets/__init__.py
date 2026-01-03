from .registry import register_dataset

from .floodnet import (
    FloodNetMask2FormerDataset,
    FloodNetSegDataset
)

#from .crarsar import CrarsarDataset

from .rescuenet import (
    RescueNetMask2FormerDataset,
    RescueNetSegDataset
)

from .rescuenet_optimized import (
    RescueNetMask2FormerDatasetOptimized
)

__all__ = [
    'register_dataset',
    'FloodNetMask2FormerDataset',
    'FloodNetSegDataset',
    'CrarsarDataset',
    'RescueNetMask2FormerDataset',
    'RescueNetSegDataset'
]
