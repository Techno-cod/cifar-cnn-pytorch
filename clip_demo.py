import clip
import torch
from PIL import Image

device = torch.device("mps")

# Load model
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"Model loaded on: {device}")
print(f"Input resolution: {model.visual.input_resolution}")

# Encode text labels
labels = ["a dog", "a cat", "an airplane", "a car", "a ship"]
text_tokens = clip.tokenize(labels).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

print(f"\nText features shape: {text_features.shape}")
print(f"Each label encoded as a {text_features.shape[1]}-dimensional vector")

import torchvision
import torchvision.transforms as transforms

# CIFAR-10 test set
cifar_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=cifar_transform)

# CIFAR-10 class descriptions as natural language
cifar_labels = [
    "a photo of an airplane",
    "a photo of a car",
    "a photo of a bird",
    "a photo of a cat",
    "a photo of a deer",
    "a photo of a dog",
    "a photo of a frog",
    "a photo of a horse",
    "a photo of a ship",
    "a photo of a truck"
]

text_tokens = clip.tokenize(cifar_labels).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Test on 1000 images
correct = 0
total = 1000

for i in range(total):
    image, label = testset[i]
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T)
        predicted = similarity.argmax().item()

    if predicted == label:
        correct += 1

accuracy = 100 * correct / total
print(f"\nCLIP zero-shot accuracy on CIFAR-10: {accuracy:.2f}%")
print(f"Correct: {correct}/{total}")
print(f"\nFor comparison:")
print(f"  Random guessing:        10.00%")
print(f"  Your scratch CNN:       72.50%")
print(f"  Your ResNet fine-tuned: 84.05%")
print(f"  CLIP zero-shot:         {accuracy:.2f}%")

import json

results = {
    "task": "CIFAR-10 image classification",
    "models": {
        "random_baseline": 10.0,
        "scratch_cnn": 72.5,
        "resnet18_finetuned": 84.05,
        "clip_zero_shot": round(accuracy, 2)
    },
    "clip_model": "ViT-B/32",
    "images_tested": total,
    "note": "CLIP achieves this with zero task-specific training"
}

with open('clip_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Results saved to clip_results.json")