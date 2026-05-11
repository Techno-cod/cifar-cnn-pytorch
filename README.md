# CIFAR-10 Image Classification — PyTorch CNN, ResNet-18 & CLIP

Three approaches to image classification on CIFAR-10, built in PyTorch on Apple M1 (MPS backend).

## Results

| Model | Approach | Test Accuracy |
|-------|----------|---------------|
| Custom CNN | Trained from scratch | 72.5% |
| ResNet-18 | Full fine-tuning (transfer learning) | 84.05% |
| CLIP ViT-B/32 | Zero-shot (no training) | 85.70% |

CLIP achieves 85.70% with **zero task-specific training** — outperforming the fine-tuned ResNet by +1.65%.

## Key insight
CLIP encodes images and text into the same 512-dimensional vector space during pretraining. At inference, it finds the text label most similar to the image — no gradient updates, no CIFAR-10 examples ever seen.

## Models

### Custom CNN (`cifar_cnn.py`)
- 2 Conv2d layers + MaxPool + 3 fully connected layers
- Trained with Adam, CrossEntropyLoss, 10 epochs
- Loss: 1.35 → 0.08

### ResNet-18 Fine-tuned (`resnet_finetune.py`)
- Pretrained on ImageNet, fully fine-tuned on CIFAR-10
- Differential learning rates: 1e-4 (backbone) / 1e-3 (head)
- Data augmentation: RandomHorizontalFlip, RandomCrop
- LR scheduler: StepLR (decay ×0.5 every 3 epochs)

### CLIP Zero-Shot (`clip_demo.py`)
- Model: ViT-B/32 (Vision Transformer)
- No training — classifies using natural language prompts
- Example prompt: "a photo of a cat"
- Tested on 1,000 CIFAR-10 test images

## How to run
```bash
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
python3 cifar_cnn.py        # scratch CNN
python3 resnet_finetune.py  # fine-tuned ResNet
python3 clip_demo.py        # CLIP zero-shot
```
