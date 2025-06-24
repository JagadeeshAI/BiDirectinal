import os
import sys
import json
import logging
from datetime import datetime

import torch
from tqdm import tqdm
from torchvision.ops import box_iou
from torch.cuda.amp import autocast, GradScaler

from uitls import get_model, print_parameter_stats
from data import get_detection_loader

# -------------------- Config -------------------- #
class Config:
    # ===== General Hyperparameters =====
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    PIN_MEMORY = True

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    EPOCHS = 20

    BASE_PATH = "/media/jag/volD/BID_DATA/obd/final_OBD"
    TRAIN_DATA_PATH        = os.path.join(BASE_PATH, "train")
    VAL_DATA_PATH          = os.path.join(BASE_PATH, "val")
    TRAIN_ANNOTATION_PATH  = os.path.join(BASE_PATH, "annotations", "lvis_single_object_train.json")
    VAL_ANNOTATION_PATH    = os.path.join(BASE_PATH, "annotations", "lvis_single_object_val.json")

# -------------------- Logger -------------------- #
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

    if len(logger.handlers) == 0:
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger

# -------------------- Checkpoint -------------------- #
def save_checkpoint(model, optimizer, epoch, path="checkpoints/checkpoint.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch":                epoch,
    }, path)

# -------------------- Training -------------------- #
def detection_loss(pred_boxes, pred_logits, gt_boxes, gt_labels, num_classes):
    """
    pred_boxes: [B, 4]
    pred_logits: [B, num_classes]
    gt_boxes: [B, 4]
    gt_labels: [B]
    """
    # Box loss: Smooth L1
    loss_bbox = torch.nn.functional.smooth_l1_loss(pred_boxes, gt_boxes, reduction="mean")
    # Class loss: Cross entropy
    loss_cls = torch.nn.functional.cross_entropy(pred_logits, gt_labels, reduction="mean")
    return loss_cls, loss_bbox

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, logger, scaler):
    model.train()
    running_loss = 0.0
    correct_cls = 0
    total_cls = 0
    pbar = tqdm(loader, desc=f"🚀 Epoch {epoch}", leave=False)

    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        with autocast():
            outputs = model(images, targets)
            if isinstance(outputs, dict):
                loss_cls = outputs.get("loss_cls", torch.tensor(0.0, device=device))
                loss_bbox = outputs.get("loss_bbox", torch.tensor(0.0, device=device))
                total_loss = loss_cls + loss_bbox
                loss_dict = {"loss_cls": loss_cls, "loss_bbox": loss_bbox}
            else:
                total_loss = outputs
                loss_dict = {"total_loss": total_loss}

        scaler.scale(total_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += total_loss.item()

        # Compute accuracy for this batch
        with torch.no_grad():
            preds = model(images)
            for gt, out in zip(targets, preds):
                pred_labels = out["labels"]
                gt_labels = gt["labels"]
                # Both should be (Q,) or (1,)
                if pred_labels.dim() == 0:
                    pred_labels = pred_labels.unsqueeze(0)
                if gt_labels.dim() == 0:
                    gt_labels = gt_labels.unsqueeze(0)
                # For single-object, just compare first
                correct_cls += (pred_labels[0] == gt_labels[0]).item()
                total_cls += 1

        postfix = {k: f"{v.item():.4f}" for k, v in loss_dict.items() if isinstance(v, torch.Tensor)}
        postfix["total"] = f"{total_loss.item():.4f}"
        postfix["acc"] = f"{(correct_cls/total_cls)*100:.2f}%" if total_cls > 0 else "0.00%"
        pbar.set_postfix(postfix)

    avg_loss = running_loss / len(loader)
    accuracy = correct_cls / total_cls if total_cls > 0 else 0.0
    logger.info(f"✅ Epoch {epoch} – Avg Train Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
    return avg_loss, accuracy

# -------------------- Validation -------------------- #
@torch.no_grad()
def validate(model, loader, device, logger):
    model.eval()
    detected = set()
    all_ious = []
    all_l1   = []
    total_loss = 0.0
    correct_cls = 0
    total_cls = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images, targets)
        if isinstance(outputs, dict):
            loss_cls = outputs.get("loss_cls", torch.tensor(0.0, device=device))
            loss_bbox = outputs.get("loss_bbox", torch.tensor(0.0, device=device))
            total_loss += (loss_cls + loss_bbox).item()
        else:
            preds = outputs
            pred_boxes = torch.stack([out["boxes"] for out in preds]).to(device)
            pred_logits = torch.stack([out["scores"] for out in preds]).to(device)
            gt_boxes = torch.stack([t["boxes"] for t in targets]).to(device)
            gt_labels = torch.stack([t["labels"] for t in targets]).to(device)
            loss_cls = torch.nn.functional.cross_entropy(pred_logits, gt_labels)
            loss_bbox = torch.nn.functional.smooth_l1_loss(pred_boxes, gt_boxes)
            total_loss += (loss_cls + loss_bbox).item()

        # For metrics, get predictions
        preds = model(images)
        for gt, out in zip(targets, preds):
            pred_box = out["boxes"].view(1, 4)
            gt_box = gt["boxes"].view(1, 4)

            if pred_box.shape[-1] != 4 or gt_box.shape[-1] != 4:
                continue
            try:
                iou = box_iou(pred_box, gt_box)[0, 0].item()
                l1_error = torch.abs(pred_box - gt_box).mean().item()
                all_ious.append(iou)
                all_l1.append(l1_error)
            except Exception as e:
                logger.warning(f"⚠️ Skipping box comparison: {e}")
                continue

            pred_labels = out["labels"]
            gt_labels = gt["labels"]
            if pred_labels.dim() == 0:
                pred_labels = pred_labels.unsqueeze(0)
            if gt_labels.dim() == 0:
                gt_labels = gt_labels.unsqueeze(0)
            correct_cls += (pred_labels[0] == gt_labels[0]).item()
            total_cls += 1

            labels = out["labels"].cpu()
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            detected.update(labels.tolist())

    avg_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
    avg_l1 = sum(all_l1) / len(all_l1) if all_l1 else 0.0
    avg_val_loss = total_loss / len(loader)
    accuracy = correct_cls / total_cls if total_cls > 0 else 0.0

    metrics = {
        "num_classes_detected": len(detected),
        "classes_detected":     sorted(detected),
        "avg_iou":              avg_iou,
        "avg_l1_error":         avg_l1,
        "val_loss":             avg_val_loss,
        "accuracy":             accuracy,
    }
    logger.info(f"📊 Validation Metrics: {metrics}")
    return metrics

# -------------------- Main Entry -------------------- #
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(log_dir="logs")
    logger.info(f"Using device: {device}")
    logger.info(f"Batch size: {Config.BATCH_SIZE}, Epochs: {Config.EPOCHS}")

    train_loader = get_detection_loader(mode="train", batch_size=Config.BATCH_SIZE)
    val_loader   = get_detection_loader(mode="val", batch_size=Config.BATCH_SIZE)

    model, _ = get_model(
        use_lora=False,
        msa=[1, 0, 1],
        model_type="detector",
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

    scaler = GradScaler()
    best_iou = 0.0

    for epoch in range(1, Config.EPOCHS + 1):
        avg_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer,
            scheduler, device, epoch, logger, scaler
        )

        val_metrics = validate(model, val_loader, device, logger)
        score = val_metrics["avg_iou"]

        if score > best_iou:
            best_iou = score
            save_checkpoint(model, optimizer, epoch, path="checkpoints/best.pth")
            logger.info(f"🎉 New best model at epoch {epoch} (Avg IoU: {score:.4f}, Accuracy: {val_metrics['accuracy']*100:.2f}%)")

        ckpt_path   = f"checkpoints/epoch_{epoch}.pth"
        metric_path = f"checkpoints/metrics_epoch_{epoch}.json"
        save_checkpoint(model, optimizer, epoch, path=ckpt_path)
        with open(metric_path, "w") as f:
            json.dump(val_metrics, f, indent=2)
        logger.info(f"Saved checkpoint → {ckpt_path}")
        logger.info(f"Saved metrics   → {metric_path}")
        logger.info(f"Epoch {epoch} complete! Avg Loss: {avg_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Val Acc: {val_metrics['accuracy']*100:.2f}%, Avg IoU: {score:.4f}")

    logger.info("🏆 Training complete!")
    logger.info(f"Best Avg IoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()