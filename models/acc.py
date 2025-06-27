import torch
import torch.nn as nn
import json
from collections import defaultdict
from data import get_dynamic_loader
from utils.uitls import get_model, print_parameter_stats
from tqdm import tqdm

# ---- Config ----
CR = (0, 99)
BS = 32
NW = 4
CHECKPOINT = "classifier_0_49_best.pth"
RESULTS_FILE = "results.json"

def evaluate_model(model, dataloader, device, num_classes):
    model.eval()
    total, correct = 0, 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="🔍 Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            for label, pred in zip(labels, preds):
                class_total[label.item()] += 1
                if label.item() == pred.item():
                    class_correct[label.item()] += 1

    class_acc = {}
    for i in range(num_classes):
        if class_total[i] == 0:
            acc = None  # No samples for this class
        else:
            acc = 100.0 * class_correct[i] / class_total[i]
        class_acc[f"class_{i}"] = acc

    overall_acc = 100.0 * correct / total

    print(f"\n✅ Accuracy on validation set ({total} samples): {overall_acc:.2f}%")

    return {
        "overall_accuracy": round(overall_acc, 2),
        "classwise_accuracy": class_acc
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")

    val_loader = get_dynamic_loader(class_range=CR, mode="val", batch_size=BS, num_workers=NW)

    model, args = get_model(class_range=CR, use_lora=False, msa=[1, 0, 1])
    state_dict = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)

    print_parameter_stats(model)
    result = evaluate_model(model, val_loader, device, num_classes=CR[1] - CR[0] + 1)

    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=4)
    print(f"\n📁 Saved results to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
