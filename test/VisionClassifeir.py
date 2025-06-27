import torch
from torchvision import transforms
from PIL import Image
from utils.uitls import print_parameter_stats, get_model, apply_lora, freeze_output

# --- Only run for classifier ---
model_type = 'classifier'
img_size = 224  # classifier uses 224

# Define the transform
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

model, args = get_model(class_range=(10, 49),use_lora=True, msa=[1, 0, 1])
model = apply_lora(model, args)

model.eval()
print_parameter_stats(model)

dummy_image = Image.new("RGB", (img_size, img_size), (255, 255, 255))
input_tensor = transform(dummy_image).unsqueeze(0).to(args._device)  # (B, C, H, W)

with torch.no_grad():
    output = model(input_tensor)

print(f"✅ {model_type.capitalize()} Output Shape:", output.shape)
print("📌 First 5 dims:", output[0][:5].cpu().numpy())
