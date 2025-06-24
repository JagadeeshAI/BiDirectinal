import os
import sys
import json
import logging
from datetime import datetime

import torch
from tqdm import tqdm
from torchvision.ops import box_iou

from config import Config
from utils.uitls import get_model, print_parameter_stats
from models.data import get_detection_loader


def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_{timestamp}.log")

    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def save_checkpoint(model, optimizer, epoch, path="checkpoints/checkpoint.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch":                epoch,
    }, path)


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, logger):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"🚀 Epoch {epoch}", leave=False)

    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # Forward pass
        loss_dict = model(images, targets)
        loss_cls = loss_dict["loss_cls"]
        loss_bbox = loss_dict["loss_bbox"]

        # Weighted total loss
        total_loss = loss_cls + 0.1 * loss_bbox

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += total_loss.item()

        postfix = {
            "loss_cls": f"{loss_cls.item():.4f}",
            "loss_bbox": f"{loss_bbox.item():.4f}",
            "total": f"{total_loss.item():.4f}",
        }
        pbar.set_postfix(postfix)

    avg_loss = running_loss / len(loader)
    logger.info(f"✅ Epoch {epoch} – Avg Loss: {avg_loss:.4f}")
    return avg_loss



@torch.no_grad()
def validate(model, loader, device, logger):
    model.eval()
    detected = set()
    all_ious = []
    all_l1   = []

    for images, targets in loader:
        images = [img.to(device) for img in images]
        gt_boxes = [t["boxes"].to(device) for t in targets]

        outputs = model(images)
        for gt, out in zip(gt_boxes, outputs):
            if out["boxes"].numel() != 4:
                continue  # prediction must be exactly 1 box [4]

            pred_box = out["boxes"].view(1, 4)  # [1, 4]
            gt_box   = gt.view(1, 4)            # [1, 4]

            try:
                iou = box_iou(pred_box, gt_box)[0, 0].item()
                l1_error = torch.abs(pred_box - gt_box).mean().item()
                all_ious.append(iou)
                all_l1.append(l1_error)
            except Exception as e:
                logger.warning(f"⚠️ Skipping box comparison: {e}")
                continue

            labels = out["labels"].cpu()
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            detected.update(labels.tolist())

    avg_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
    avg_l1 = sum(all_l1) / len(all_l1) if all_l1 else 0.0

    metrics = {
        "num_classes_detected": len(detected),
        "classes_detected":     sorted(detected),
        "avg_iou":              avg_iou,
        "avg_l1_error":         avg_l1,
    }
    logger.info(f"📊 Validation Metrics: {metrics}")
    return metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(log_dir="logs")
    logger.info(f"Using device: {device}")
    logger.info(f"Batch size: {Config.BATCH_SIZE}, Epochs: {Config.EPOCHS}")

    train_loader = get_detection_loader(mode="train", batch_size=Config.BATCH_SIZE)
    val_loader   = get_detection_loader(mode="val",   batch_size=Config.BATCH_SIZE)

    model, _ = get_model(
        use_lora=False,
        msa=[1, 0, 1],
        model_type="face",
    )
    model = model.to(device)
    print_parameter_stats(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=0.05,
    )
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        total_steps=Config.EPOCHS * steps_per_epoch,
        pct_start=0.1,
        div_factor=100,
    )

    best_score = 0.0

    for epoch in range(1, Config.EPOCHS + 1):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer,
            scheduler, device,
            epoch, logger
        )

        val_metrics = validate(model, val_loader, device, logger)
        score = val_metrics["num_classes_detected"]

        if score > best_score:
            best_score = score
            save_checkpoint(model, optimizer, epoch, path="checkpoints/best.pth")
            logger.info(f"🎉 New best model at epoch {epoch} (detected {score} classes)")

        ckpt_path   = f"checkpoints/epoch_{epoch}.pth"
        metric_path = f"checkpoints/metrics_epoch_{epoch}.json"
        save_checkpoint(model, optimizer, epoch, path=ckpt_path)
        with open(metric_path, "w") as f:
            json.dump(val_metrics, f, indent=2)
        logger.info(f"Saved checkpoint → {ckpt_path}")
        logger.info(f"Saved metrics   → {metric_path}")
        logger.info(f"Epoch {epoch} complete! Avg Loss: {avg_loss:.4f}, Detected Classes: {score}")

    logger.info("🏆 Training complete!")
    logger.info(f"Best #classes detected: {best_score}")


if __name__ == "__main__":
    main()
