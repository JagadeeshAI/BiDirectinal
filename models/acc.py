import os
import torch
import json
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

from utils.uitls import get_model
from models.data import get_dynamic_loader
from models.train import ArcFaceHead
from config import Config


def compute_accuracy(backbone, arc_head, dataloader, device, total_classes=100):
    backbone.eval()
    arc_head.eval()

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="🔍 Evaluating"):
            images, labels = images.to(device), labels.to(device)
            embeddings = backbone(images)

            # ✅ Remove label-based computation here
            cosine = F.linear(F.normalize(embeddings), F.normalize(arc_head.weight))
            preds = torch.argmax(cosine, dim=1)

            for pred, true in zip(preds, labels):
                class_total[int(true)] += 1
                if pred == true:
                    class_correct[int(true)] += 1

    # Compute accuracies
    classwise_accuracy = {}
    total_correct = 0
    total_seen = 0

    for i in range(total_classes):
        correct = class_correct.get(i, 0)
        total = class_total.get(i, 0)
        acc = 100.0 * correct / total if total > 0 else 0.0
        classwise_accuracy[f"class_{i}"] = acc
        total_correct += correct
        total_seen += total

    overall_acc = 100.0 * total_correct / total_seen if total_seen > 0 else 0.0

    return overall_acc, classwise_accuracy


def main():
    device = Config.DEVICE
    model, _ = get_model(use_lora=True)
    arc_head = ArcFaceHead(in_features=768, out_features=100).to(device)

    # Load best checkpoint
    ckpt_path = os.path.join(Config.CKPT_DIR, "best_model_epoch3.pth")  
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["backbone"])
    arc_head.load_state_dict(ckpt["arcface_head"])

    val_loader = get_dynamic_loader(
        Config.TEST_DATA_PATH, range(100), mode="val", batch_size=Config.BATCH_SIZE
    )

    overall_acc, classwise_acc = compute_accuracy(model, arc_head, val_loader, device)

    results = {
        "overall_accuracy": round(overall_acc, 2),
        "classwise_accuracy": {k: round(v, 2) for k, v in classwise_acc.items()}
    }

    os.makedirs("results", exist_ok=True)
    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("✅ Evaluation complete. Saved to results/results.json")


if __name__ == "__main__":
    main()
