import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError, ImageFile
from config import Config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# allow loading of truncated images without error
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------- Dataset -------------------- #
class LvisDetectionDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        annotation_path: str,
        transforms: transforms.Compose = None,
        class_range: tuple[int,int] = (0, 99),  
    ):
        self.image_dir = image_dir
        self.transforms = transforms
        self.min_id, self.max_id = class_range

        # Load annotation JSON
        with open(annotation_path, "r") as f:
            data = json.load(f)

        # Map image_id -> file_name
        self.image_id_to_filename = {
            img["id"]: img.get("file_name", f"{img['id']:012}.jpg")
            for img in data["images"]
        }

        # Collect annotations by image, filtering by category_id
        self.annotations: dict[int, list] = {}
        for ann in data["annotations"]:
            cid = ann["category_id"]
            if cid < self.min_id or cid > self.max_id:
                continue
            img_id = ann["image_id"]
            self.annotations.setdefault(img_id, []).append(ann)

        # Build list of valid image_ids
        self.image_ids = []
        for img_id, anns in self.annotations.items():
            # Use first annotation to locate file
            ann = anns[0]
            cid = ann["category_id"]
            fname = self.image_id_to_filename[img_id]
            img_path = os.path.join(self.image_dir, str(cid), fname)

            # Skip if file missing
            if not os.path.exists(img_path):
                continue
            # Verify image, delete if corrupted
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError) as e:
                print(f"Warning: corrupted image '{img_path}', deleting: {e}")
                try:
                    os.remove(img_path)
                except OSError as rm_err:
                    print(f"Error deleting corrupted file {img_path}: {rm_err}")
                continue

            self.image_ids.append(img_id)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        fname = self.image_id_to_filename[image_id]
        ann = self.annotations[image_id][0]
        cid = ann["category_id"]
        img_path = os.path.join(self.image_dir, str(cid), fname)

        # Load original image
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Normalize bbox to [0,1] in original image space
        x, y, w, h = ann["bbox"]
        x1 = x / orig_w
        y1 = y / orig_h
        x2 = (x + w) / orig_w
        y2 = (y + h) / orig_h

        # Apply transforms (resize, to tensor, normalize)
        if self.transforms:
            image = self.transforms(image)

        # Convert normalized coords to pixel coords in transformed image
        S = Config.IMAGE_SIZE
        x1 *= S
        y1 *= S
        x2 *= S
        y2 *= S

        target = {
            "boxes":    torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32),
            "labels":   torch.tensor([cid], dtype=torch.int64),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
        }
        return image, target


# -------------------- Transforms -------------------- #
def get_transforms(mode: str) -> transforms.Compose:
    base = [
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.MEAN, std=Config.STD),
    ]
    if mode == "train":
        return transforms.Compose([transforms.RandomHorizontalFlip(), *base])
    return transforms.Compose(base)


# -------------------- DataLoader Factory -------------------- #
def get_detection_loader(
    image_dir: str       = None,
    annotation_path: str = None,
    batch_size: int      = 4,
    mode: str            = "train",
    num_workers: int     = Config.NUM_WORKERS,
    pin_memory: bool     = Config.PIN_MEMORY,
    drop_last: bool      = False,
    class_range: tuple[int,int] = (0, 49),
) -> DataLoader:
    if mode == "val":
        image_dir      = image_dir      or Config.VAL_DATA_PATH
        annotation_path = annotation_path or Config.VAL_ANNOTATION_PATH
    else:
        image_dir      = image_dir      or Config.TRAIN_DATA_PATH
        annotation_path = annotation_path or Config.TRAIN_ANNOTATION_PATH

    dataset = LvisDetectionDataset(
        image_dir=image_dir,
        annotation_path=annotation_path,
        transforms=get_transforms(mode),
        class_range=class_range,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )


if __name__ == "__main__":
 

    def load_category_map(annotation_path):
        with open(annotation_path, "r") as f:
            data = json.load(f)
        return {cat["id"]: cat["name"] for cat in data.get("categories", [])}

    def visualize(image, target, cat_map):
        # Convert image tensor to HWC numpy and unnormalize
        image = image.permute(1, 2, 0).numpy()
        mean = np.array(Config.MEAN)
        std = np.array(Config.STD)
        image = image * std + mean
        image = image.clip(0, 1)

        fig, ax = plt.subplots(1)
        ax.imshow(image)

        box = target["boxes"][0]  # [x1, y1, x2, y2]
        label = target["labels"].item()
        label_name = cat_map.get(label, f"Class {label}")

        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        ax.text(x1, y1 - 5, label_name, fontsize=12, color='yellow',
                backgroundcolor='black')

        plt.axis("off")
        plt.tight_layout()
        plt.show()


    # Load validation sample
    loader = get_detection_loader(mode="val", batch_size=1)
    category_map = load_category_map(Config.VAL_ANNOTATION_PATH)

    image, target = next(iter(loader))
    image = image[0]
    target = target[0]

    print(f"🖼️ Image ID: {target['image_id'].item()} | Category ID: {target['labels'].item()}")
    visualize(image, target, category_map)
