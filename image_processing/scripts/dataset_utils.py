from typing import Dict, List, Sequence, Tuple

import torch


def semantic_map_to_targets(
    semantic_map: torch.Tensor,
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    class_ids = torch.unique(semantic_map)
    valid_class_ids = class_ids[class_ids != ignore_index]

    if valid_class_ids.numel() == 0:
        empty_class_ids = torch.zeros((0, ), dtype=torch.long)
        empty_class_masks = torch.zeros(
            (0, semantic_map.shape[0], semantic_map.shape[1]),
            dtype=torch.float32,
        )
        return empty_class_ids, empty_class_masks

    class_masks: List[torch.Tensor] = []
    for class_id in valid_class_ids:
        class_masks.append((semantic_map == class_id).to(torch.float32))

    return valid_class_ids.to(torch.long), torch.stack(class_masks, dim=0)


class GooseMask2FormerCollator:

    def __init__(self, ignore_index: int):
        self.ignore_index = ignore_index

    def __call__(
        self, batch: Sequence[Tuple[torch.Tensor,
                                    torch.Tensor]]) -> Dict[str, object]:
        images, semantic_maps = zip(*batch)
        class_labels: List[torch.Tensor] = []
        mask_labels: List[torch.Tensor] = []

        for semantic_map in semantic_maps:
            classes, masks = semantic_map_to_targets(semantic_map,
                                                     self.ignore_index)
            class_labels.append(classes)
            mask_labels.append(masks)

        return {
            "pixel_values": torch.stack(images),
            "semantic_maps": torch.stack(semantic_maps),
            "class_labels": class_labels,
            "mask_labels": mask_labels,
        }


def move_batch_to_device(batch: Dict[str, object],
                         device: torch.device) -> Dict[str, object]:
    return {
        "pixel_values": batch["pixel_values"].to(device, non_blocking=True),
        "semantic_maps": batch["semantic_maps"].to(device, non_blocking=True),
        "class_labels": [item.to(device) for item in batch["class_labels"]],
        "mask_labels": [item.to(device) for item in batch["mask_labels"]],
    }
