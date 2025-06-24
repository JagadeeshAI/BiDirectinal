import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from backbone.vit_bid import VisionClassifier, PatchEmbed  # Adjust import as needed
import matplotlib.pyplot as plt
import os

# ---- 1. Define your tuning_config ----
class TuningConfig:
    def __init__(self, device, embed_dim=768, attn_bn=8):
        self.ffn_adapt = True
        self.ffn_num = 8
        self.ffn_adapter_init_option = "lora"
        self.ffn_adapter_scalar = 1.0
        self.ffn_adapter_layernorm_option = "in"
        self.msa_adapt = True
        self.msa = [1, 0, 1]
        self.general_pos = []
        self.specfic_pos = list(range(12))  # All blocks
        self.use_distillation = False
        self.use_block_weight = False
        self.vpt_on = False
        self.vpt_num = 0
        self.task_type = "classification"
        self._device = device
        # Add these attributes for compatibility with vit_bid.py:
        self.d_model = embed_dim
        self.attn_bn = attn_bn
        self.random_orth = False  # Set True if you want orthogonal init

# ---- 2. Data transforms and loaders ----
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Load the full dataset
train_dataset_full = ImageFolder('data/train', transform=transform_train)
val_dataset_full = ImageFolder('data/val', transform=transform_val)

# Get indices for samples belonging to the first 50 classes
first_50_classes = list(range(50))
train_indices = [i for i, (_, label) in enumerate(train_dataset_full.samples) if label in first_50_classes]
val_indices = [i for i, (_, label) in enumerate(val_dataset_full.samples) if label in first_50_classes]

# Create subset datasets
train_dataset = Subset(train_dataset_full, train_indices)
val_dataset = Subset(val_dataset_full, val_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# ---- 3. Model, optimizer, loss ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 50  # Update num_classes for the model
embed_dim = 768  # or whatever you use in VisionClassifier
attn_bn = 8      # or your desired bottleneck size
tuning_config = TuningConfig(device, embed_dim=embed_dim, attn_bn=attn_bn)

model = VisionClassifier(
    global_pool=False,
    img_size=224,
    patch_size=16,
    in_chans=3,
    num_classes=num_classes,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    qkv_bias=True,
    representation_size=None,
    distilled=False,
    drop_rate=0.1,
    attn_drop_rate=0.1,
    drop_path_rate=0.1,
    embed_layer=PatchEmbed,
    norm_layer=None,
    act_layer=None,
    weight_init="",
    tuning_config=tuning_config,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

checkpoint_path = "DEFcheckpoint.pth"

# ---- Resume logic ----
start_epoch = 0
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    start_epoch = checkpoint['epoch'] + 1
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    train_accs = checkpoint.get('train_accs', [])
    val_accs = checkpoint.get('val_accs', [])
    print(f"Resumed from epoch {start_epoch}")

# ---- 4. Training and validation loops ----
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        #print(f"Input batch shape: {imgs.shape}")  # Print input shape
        optimizer.zero_grad()
        outputs = model(imgs)
        #print(f"Model output shape: {outputs.shape}")  # Print output shape
        loss = criterion(outputs, labels)
        #print(f"Loss tensor shape: {loss.shape if hasattr(loss, 'shape') else 'scalar'}")
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        #print(f"Predictions shape: {preds.shape}")  # Print predictions shape
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            #print(f"Validation input batch shape: {imgs.shape}")  # Print input shape
            outputs = model(imgs)
            #print(f"Validation output shape: {outputs.shape}")  # Print output shape
            loss = criterion(outputs, labels)
            #print(f"Validation loss tensor shape: {loss.shape if hasattr(loss, 'shape') else 'scalar'}")
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            #print(f"Validation predictions shape: {preds.shape}")  # Print predictions shape
            correct += preds.eq(labels).sum().item()
            total += imgs.size(0)
    return running_loss / total, correct / total

# ---- 5. Main training loop with curve plotting ----
num_epochs = 30
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(start_epoch, num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    scheduler.step()
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
    }, checkpoint_path)

    # Optionally save model for each epoch
    torch.save(model.state_dict(), f"vit_vgg_epoch{epoch+1}.pth")
    torch.cuda.empty_cache()  # <--- Add this line

# ---- 6. Plot curves ----
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1, num_epochs+1), train_accs, label='Train Acc')
plt.plot(range(1, num_epochs+1), val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()


import json

config = {
    "prefix": "",
    "dataset": "vgg2_224",
    "memory_size": 0,
    "memory_per_class": 0,
    "fixed_memory": False,
    "shuffle": True,
    "init_cls": 50,  # You are using first 50 classes
    "increment": 0,  # No incremental learning in this script
    "model_name": "vit_lora",
    "backbone_type": "vit_base_patch16_224_lora",
    "device": ["cuda" if torch.cuda.is_available() else "cpu"],
    "seed": [1993],
    "init_epochs": num_epochs,
    "init_lr": optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else 3e-4,
    "later_epochs": num_epochs,
    "later_lr": optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else 3e-4,
    "batch_size": train_loader.batch_size,
    "weight_decay": optimizer.param_groups[0]['weight_decay'] if hasattr(optimizer, 'param_groups') else 0.05,
    "min_lr": 0,
    "optimizer": "adamw",
    "scheduler": "step",
    "pretrained": False,
    "vpt_type": "None",
    "prompt_token_num": 0,
    "ffn_num": tuning_config.ffn_num,
    "use_diagonal": False,
    "recalc_sim": False,
    "alpha": 0.0,
    "use_init_ptm": False,
    "beta": 0,
    "use_old_data": False,
    "use_reweight": False,
    "moni_adam": False,
    "adapter_num": -1,
    "msa_adapt": tuning_config.msa_adapt,
    "use_distillation": tuning_config.use_distillation,
    "use_block_weight": tuning_config.use_block_weight,
    "msa": tuning_config.msa,
    "general_pos": tuning_config.general_pos,
    "specfic_pos": tuning_config.specfic_pos
}

with open("define_vgg2_config.json", "w") as f:
    json.dump(config, f, indent=4)