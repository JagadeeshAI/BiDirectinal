import torch
from torchvision import transforms
from PIL import Image
from utils.uitls import print_parameter_stats, get_model, apply_lora

# ----------------- Load VisionDetector -----------------
model, args = get_model(use_lora=True, msa=[1, 0, 1], model_type="detector")  # model_type selects VisionDetector
model = apply_lora(model, args)

model.eval()
print_parameter_stats(model)

# ----------------- Dummy Image for Detection -----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Use 224x224 for object detection
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

dummy_image = Image.new("RGB", (224, 224), (128, 128, 128))  # gray image
input_tensor = transform(dummy_image).unsqueeze(0).to(args._device)

# ----------------- Inference -----------------
with torch.no_grad():
    outputs = model(input_tensor)  # returns dict: boxes, scores, labels

# ----------------- Print Results -----------------
print("✅ Detection Output:")
print("📦 Boxes:", outputs['boxes'].cpu().numpy())
print("🏷️ Labels:", outputs['labels'].cpu().numpy())
print("🔢 Scores:", outputs['scores'][0][:5].cpu().numpy())  # print top-5 scores of first item
