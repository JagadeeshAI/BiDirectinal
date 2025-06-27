import torch
import torch.nn as nn
from data import get_dynamic_loader
from utils.uitls import get_model, print_parameter_stats,apply_lora
from tqdm import tqdm

# ---- Config ----
CR = (0, 49)
BS = 32
NW = 4
EPOCHS = 50
LR = 5e-5
EARLY_STOPPING_PATIENCE = 10
USE_LORA = False  

def train_one_epoch(model, loader, crit, opt, dev, ep):
    model.train()
    t_loss, t_corr, t_num = 0, 0, 0
    loop = tqdm(enumerate(loader), total=len(loader), desc=f"Ep{ep+1} [TRN]", ncols=175)

    for _, (x, y) in loop:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()

        bs = y.size(0)
        t_loss += loss.item() * bs
        _, p = out.max(1)
        t_corr += p.eq(y).sum().item()
        t_num += bs

        loop.set_postfix(EL=f"{t_loss / t_num:.2f}", EA=f"{t_corr / t_num:.2f}")
    return t_loss / t_num, t_corr / t_num


def eval_one_epoch(model, loader, crit, dev, ep):
    model.eval()
    v_loss, v_corr, v_num = 0, 0, 0
    loop = tqdm(enumerate(loader), total=len(loader), desc=f"Ep{ep+1} [VAL]", ncols=175)

    with torch.no_grad():
        for _, (x, y) in loop:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            loss = crit(out, y)

            bs = y.size(0)
            v_loss += loss.item() * bs
            _, p = out.max(1)
            v_corr += p.eq(y).sum().item()
            v_num += bs

            loop.set_postfix(BL=f"{loss.item():.2f}", BA=f"{(p.eq(y).float().mean().item()):.2f}",
                             EL=f"{v_loss/v_num:.2f}", EA=f"{v_corr/v_num:.2f}")
    return v_loss / v_num, v_corr / v_num


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {dev}")

    trn_loader = get_dynamic_loader(class_range=CR, mode="train", batch_size=BS, num_workers=NW)
    val_loader = get_dynamic_loader(class_range=CR, mode="val", batch_size=BS, num_workers=NW)

    model, args = get_model(class_range=CR, use_lora=True, msa=[1, 0, 1])
    model = model.to(dev)
    print_parameter_stats(model)

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_acc = 0.0
    epochs_no_improve = 0

    for ep in range(EPOCHS):
        trn_loss, trn_acc = train_one_epoch(model, trn_loader, crit, opt, dev, ep)
        val_loss, val_acc = eval_one_epoch(model, val_loader, crit, dev, ep)
        scheduler.step()

        acc_gap = abs(trn_acc - val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"classifier_{CR[0]}_{CR[1]}_best.pth")
            print(f"✅ Ep{ep+1:02d}: New BEST VAL ACC: {val_acc:.3f} (saved)")
        else:
            epochs_no_improve += 1
            print(f"Ep{ep+1:02d}: Val Acc: {val_acc:.3f} (best: {best_acc:.3f})")

        print(f"📊 Ep{ep+1:02d} | TRN Loss: {trn_loss:.3f}, Acc: {trn_acc:.3f} | "
              f"VAL Loss: {val_loss:.3f}, Acc: {val_acc:.3f} | BEST: {best_acc:.3f} | GAP: {acc_gap:.3f}\n")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print("⏹️ Early stopping triggered (no improvement).")
            break
        if acc_gap < 0.05 and val_acc > 0.85:
            print("✅ Training converged: accuracy gap < 5% and high val accuracy.")
            break
        # if acc_gap > 0.05:
        #     print(f"❌ Accuracy gap too large ({acc_gap:.3f}) — stopping early.")
        #     break


    print("🎉 Training Complete.")


if __name__ == "__main__":
    main()
