import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm  # more robust across IDEs and terminals

from data import get_dynamic_loader
from utils.uitls import get_model, print_parameter_stats, apply_lora

# ---- Config ----
CR_OLD = (0, 49)
CR_NEW = (50, 59)
CR_ALL = (0, 59)
CR_REHARSAL = (0, 49)
BS = 16
NW = 4
EPOCHS = 30
LR = 5e-5
CHECKPOINT = "classifier_0_49_best.pth"
USE_LORA = True
LAMBDA_KD = 5.0
LAMBDA_ORTH = 0.0001


# ----- Loss Functions -----
def knowledge_distillation_loss(old_feats, new_feats, temperature=2.0):
    soft_old = nn.functional.softmax(old_feats / temperature, dim=1)
    log_new = nn.functional.log_softmax(new_feats / temperature, dim=1)
    return nn.functional.kl_div(log_new, soft_old, reduction="batchmean") * (temperature ** 2)


def orthogonality_loss(Ut_list):
    loss = 0.0
    for i in range(len(Ut_list)):
        for j in range(i + 1, len(Ut_list)):
            loss += torch.norm(torch.matmul(Ut_list[i].T, Ut_list[j]), p="fro") ** 2
    return loss


def gradient_reassignment(current_As, prev_As):
    if prev_As is None:
        return None
    with torch.no_grad():
        importance = torch.norm(prev_As, p=2, dim=1, keepdim=True)
        norm_factor = importance.sum().clamp(min=1e-6)
        scaling = importance * current_As.size(0) / norm_factor
    return scaling


def combine_loaders(loader1, loader2):
    for (x1, y1), (x2, y2) in zip(loader1, loader2):
        x = torch.cat([x1, x2], dim=0)
        y = torch.cat([y1, y2], dim=0)
        yield x, y


# ----- Training Loop -----
def train_one_epoch(model, loader, crit, opt, dev, ep, args, shared_adapter_feats=None, prev_Uts=None, prev_As=None, loader_len=None):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    loop = tqdm(enumerate(loader), total=loader_len, desc=f"Ep{ep+1} [TRN]", ncols=175)

    for batch_idx, (x, y) in loop:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)

        if args.use_distillation and shared_adapter_feats is not None:
            with torch.no_grad():
                old_feats = shared_adapter_feats(x)
            new_feats = model.get_shared_output(x)
            kd_loss = knowledge_distillation_loss(old_feats, new_feats)
            loss += LAMBDA_KD * kd_loss

        if args.use_block_weight and prev_Uts is not None:
            current_Ut = model.get_current_block_weights()
            orth_loss = orthogonality_loss(prev_Uts + [current_Ut])
            loss += LAMBDA_ORTH * orth_loss

        loss.backward()

        if args.use_distillation and hasattr(model, "get_shared_As"):
            shared_As = model.get_shared_As()
            scaling = gradient_reassignment(shared_As, prev_As)
            if scaling is not None and shared_As.grad is not None:
                shared_As.grad.mul_(scaling)

        opt.step()

        total += y.size(0)
        total_loss += loss.item() * y.size(0)
        total_correct += (out.argmax(1) == y).sum().item()

        avg_loss = total_loss / total if total > 0 else 0
        loop.set_postfix(loss=f"{avg_loss:.4f}")

    return total_loss / total, total_correct / total


# ----- Evaluation -----
def eval_epoch(model, loader, dev):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    return accuracy_score(all_labels, all_preds) * 100


# ----- Main Entrypoint -----
def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {dev}")

    trn_loader_new = get_dynamic_loader(class_range=CR_NEW, mode="train", batch_size=BS, num_workers=NW)
    reharsal_loader = get_dynamic_loader(class_range=CR_REHARSAL, mode="train", batch_size=BS, num_workers=NW)
    val_loader_old = get_dynamic_loader(class_range=CR_OLD, mode="val", batch_size=BS, num_workers=NW)
    val_loader_new = get_dynamic_loader(class_range=CR_NEW, mode="val", batch_size=BS, num_workers=NW)
    val_loader_all = get_dynamic_loader(class_range=CR_ALL, mode="val", batch_size=BS, num_workers=NW)

    model, args = get_model(class_range=CR_ALL, use_lora=USE_LORA, msa=[1, 0, 1])
    model.load_state_dict(torch.load(CHECKPOINT, map_location=dev), strict=False)
    model = model.to(dev)
    model = apply_lora(model, args)
    print_parameter_stats(model)

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_acc_all = 0.0
    prev_Uts = []
    prev_As = None

    shared_adapter_feats = model.get_shared_output if hasattr(model, "get_shared_output") else None
    if hasattr(model, "get_shared_As"):
        prev_As = model.get_shared_As().detach().clone()

    for ep in range(EPOCHS):
        combined_loader = combine_loaders(trn_loader_new, reharsal_loader)
        combined_len = min(len(trn_loader_new), len(reharsal_loader))

        train_loss, train_acc = train_one_epoch(
            model, combined_loader, crit, opt, dev, ep,
            args, shared_adapter_feats, prev_Uts, prev_As,
            loader_len=combined_len
        )

        acc_old = eval_epoch(model, val_loader_old, dev)
        acc_new = eval_epoch(model, val_loader_new, dev)
        acc_all = eval_epoch(model, val_loader_all, dev)
        scheduler.step()

        print(f"📊 Ep{ep+1:02d} | Train Acc: {train_acc:.2f}% | 0-49: {acc_old:.2f}% | 50-59: {acc_new:.2f}% | 0-59: {acc_all:.2f}%")

        if acc_all > best_acc_all:
            best_acc_all = acc_all
            torch.save(model.state_dict(), "classifier_0_59_best.pth")
            print(f"✅ New best overall acc: {acc_all:.2f}% (saved)")

    print("🎉 Incremental Training Complete.")


if __name__ == "__main__":
    main()
