import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from utils.uitls import print_parameter_stats, get_model, apply_lora

# ----------------- Load Model + Args -----------------
model, args = get_model(use_lora=True, msa=[1, 0, 1])
model = apply_lora(model, args)

model.eval()
print_parameter_stats(model)

# ----------------- Dummy Face Image -----------------
transform = transforms.Compose(
    [
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

dummy_image = Image.new("RGB", (112, 112), (255, 255, 255))
input_tensor = transform(dummy_image).unsqueeze(0).to(args._device)

# ----------------- Inference -----------------
with torch.no_grad():
    embedding = model(input_tensor)

print("✅ Face Embedding Shape:", embedding.shape)
print("📌 First 5 dims:", embedding[0][:5].cpu().numpy())
