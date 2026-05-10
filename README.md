# CIFAR-10 Image Classification — PyTorch CNN & ResNet-18 Fine-tuning

Two approaches to image classification on CIFAR-10, built in PyTorch on Apple M1 (MPS backend).

## Results

| Model | Approach | Test Accuracy |
|-------|----------|---------------|
| Custom CNN | Trained from scratch | 72.5% |
| ResNet-18 | Full fine-tuning (transfer learning) | 84.05% |

Transfer learning improved accuracy by **+11.55%** over training from scratch.

## Models

### Custom CNN (`cifar_cnn.py`)
- 2 Conv2d layers + MaxPool + 3 fully connected layers
- Trained with Adam, CrossEntropyLoss, 10 epochs
- Loss: 1.35 → 0.08

### ResNet-18 Fine-tuned (`resnet_finetune.py`)
- Pretrained on ImageNet, fully fine-tuned on CIFAR-10
- Differential learning rates: 1e-4 (backbone) / 1e-3 (classifier head)
- Data augmentation: RandomHorizontalFlip, RandomCrop
- LR scheduler: StepLR (decay ×0.5 every 3 epochs)
- Loss: 1.15 → 0.39

## Key concepts demonstrated
- CNN architecture design from scratch
- Transfer learning and full fine-tuning
- Differential learning rates for pretrained models
- Data augmentation for generalisation
- LR scheduling
- Apple M1 MPS backend for GPU-accelerated training

## How to run
```bash
pip install torch torchvision
python3 cifar_cnn.py        # train/evaluate scratch CNN
python3 resnet_finetune.py  # evaluate fine-tuned ResNet (set TRAIN=True to retrain)
```
