import torch
import torch.nn as nn
from data import get_dynamic_loader
from utils.uitls import get_model, print_parameter_stats
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

# ---- Config ----
CR = (0, 49)
BS = 32
NW = 4
EPOCHS = 50
LR = 5e-4
WEIGHT_DECAY = 1e-4
CHECKPOINT = "classifier_0_49_best.pth"

def train_one_epoch(model, loader, crit, opt, scheduler, dev, ep):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for x, y in tqdm(loader, desc=f"Epoch {ep+1} [Train]"):
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        scheduler.step()

        total += y.size(0)
        total_loss += loss.item() * y.size(0)
        total_correct += (out.argmax(1) == y).sum().item()

    return total_loss / total, total_correct / total

def eval_one_epoch(model, loader, crit, dev, ep):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"Epoch {ep+1} [Eval]"):
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            loss = crit(out, y)

            total += y.size(0)
            total_loss += loss.item() * y.size(0)
            total_correct += (out.argmax(1) == y).sum().item()

    return total_loss / total, total_correct / total

def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {dev}")

    trn_loader = get_dynamic_loader(class_range=CR, mode="train", batch_size=BS, num_workers=NW)
    val_loader = get_dynamic_loader(class_range=CR, mode="val", batch_size=BS, num_workers=NW)

    model, args = get_model(class_range=CR, use_lora=False, msa=[1, 0, 1])
    model.load_state_dict(torch.load(CHECKPOINT, map_location=dev), strict=False)
    model = model.to(dev)

    print_parameter_stats(model)

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    total_steps = len(trn_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_acc = 0.0
    for ep in range(EPOCHS):
        trn_loss, trn_acc = train_one_epoch(model, trn_loader, crit, opt, scheduler, dev, ep)
        val_loss, val_acc = eval_one_epoch(model, val_loader, crit, dev, ep)

        print(f"📊 Epoch {ep+1:02d} | Train Acc: {trn_acc:.2f} | Val Acc: {val_acc:.2f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"finetuned_regularized_{CR[0]}_{CR[1]}.pth")
            print(f"✅ New Best Val Acc: {val_acc:.2f} (saved)")

    print("🎉 Fine-Tuning with Regularization Complete.")

if __name__ == "__main__":
    main()
