from .registry import register_dataset
from .floodnet import FloodNetSegDataset, FloodNetMask2FormerDataset

@register_dataset("crarsar_segformer")
class CRARSARSegformerDataset(FloodNetSegDataset):
    """
    Dataset class for CRARSAR dataset for SegFormer model.
    Inherits from FloodNetSegDataset since the structure is similar.
    """
    CLASSES = [ "background", "no damage", "minor damage", "major damage", "destroyed", "un-classified"]
    label2id = {"background": 0,
                "no damage": 1,
                "minor damage": 2,
                "major damage": 3,
                "destroyed": 4,
                "un-classified": 255,}
    id2label = {0: "background",
                1: "no damage",
                2: "minor damage",
                3: "major damage",
                4: "destroyed",
                255: "un-classified",}
    def __init__(
        self,
        root: str,
        split: str,
        image_processor,
        num_classes: int = 6,
        image_size: int = 512,
        augment: bool = False,
        ignore_index: int = 255,
    ):
        super().__init__(
            root=root,
            split=split,
            image_processor=image_processor,
            num_classes=num_classes,
            image_size=image_size,
            augment=augment,
            ignore_index=ignore_index,
        )
        
@register_dataset("crarsar_mask2former")
class CRARSARMask2FormerDataset(FloodNetMask2FormerDataset):
    """
    Dataset class for CRARSAR dataset for Mask2Former model.
    Inherits from FloodNetMask2FormerDataset since the structure is similar.
    """
    CLASSES = [ "background", "no damage", "minor damage", "major damage", "destroyed", "un-classified"]
    label2id = {"background": 0,
                "no damage": 1,
                "minor damage": 2,
                "major damage": 3,
                "destroyed": 4,
                "un-classified": 255,}
    id2label = {0: "background",
                1: "no damage",
                2: "minor damage",
                3: "major damage",
                4: "destroyed",
                255: "un-classified",}
    def __init__(
        self,
        root: str,
        split: str,
        image_processor,
        num_classes: int = 6,
        image_size: int = 512,
        augment: bool = False,
        ignore_index: int = 255,
    ):
        super().__init__(
            root=root,
            split=split,
            image_processor=image_processor,
            num_classes=num_classes,
            image_size=image_size,
            augment=augment,
            ignore_index=ignore_index,
        )