import albumentation as A
from albumentations.pytorch.transforms import ToTensorV2


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_train_transforms():
    return A.Compose(
        [
            A.OneOf(
                [
                    A.HueSaturationValue(
                        hue_shift_limit=0.2,
                        sat_shift_limit=0.2,
                        val_shift_limit=0.2,
                        p=0.9,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.9
                    ),
                ],
                p=0.9,
            ),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1),
            A.CoarseDropout(
                num_holes_range=(1, 8), max_height=64, max_width=64, fill_value=0, p=0.5
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="coco",
            min_area=0,
            min_visibility=0,
            label_fields=["labels"],
            clip=True,
        ),
    )


def get_valid_transforms():
    return A.Compose(
        [A.Resize(height=512, width=512, p=1.0), ToTensorV2(p=1.0)],
        p=1.0,
        bbox_params=A.BboxParams(
            format="coco",
            min_area=0,
            min_visibility=0,
            label_fields=["labels"],
            clip=True,
        ),
    )


def convert_to_detr(df):
    df["w"] = df["xmax"] - df["xmin"]
    df["h"] = df["ymax"] - df["ymin"]
    df["x"] = df["xmin"] + df["w"] / 2
    df["y"] = df["ymin"] + df["h"] / 2
    df["area"] = df["w"] * df["h"]
    df.drop(columns=["xmin", "ymin", "xmax", "ymax"], inplace=True)


def convert_to_pascal_voc(results):
    for result in results:
        boxes = result["boxes"]

    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    xmin = x - w / 2
    ymin = y - h / 2
    xmax = x + w / 2
    ymax = y + h / 2

    result["boxes"] = np.stack([xmin, ymin, xmax, ymax], axis=1)


def collate_fn(batch):
    return tuple(zip(*batch))


