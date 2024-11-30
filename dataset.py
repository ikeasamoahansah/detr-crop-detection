import torch
import numpy as np
import cv2
import albumentations as A
from torch.utils.data import Dataset

DIR_TRAIN = "/content/dataset/images"


class CropDataset(Dataset):
    def __init__(self, image_ids, dataframe, class_map, transforms=None):
        self.image_ids = image_ids
        self.df = dataframe
        self.transforms = transforms
        self.class_map = class_map

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        records = self.df[self.df["Image_ID"] == image_id]

        image = cv2.imread(f"{DIR_TRAIN}/{image_id}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # DETR takes in data in coco format
        boxes = records[["x", "y", "w", "h"]].values

        # Area of bb
        area = self.df["area"]

        labels = [
            self.class_map.inverse(class_name)
            for class_name in self.df["class"].tolist()
        ]

        if self.transforms:
            sample = {"image": image, "bboxes": boxes, "labels": torch.tensor(labels)}
            sample = self.transforms(**sample)
            image = sample["image"]
            boxes = sample["bboxes"]
            labels = sample["labels"]

        # Normalizing BBOXES

        _, h, w = image.shape
        boxes = A.core.bbox_utils.normalize_bboxes(sample["bboxes"], (h, w))
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.long)
        target["image_id"] = torch.tensor([index])
        target["area"] = area

        return image, target, image_id
