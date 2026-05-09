# CIFAR-10 Image Classifier — PyTorch CNN

A convolutional neural network built from scratch in PyTorch, trained on the CIFAR-10 dataset.

## Result
**72.5% test accuracy** on 10,000 unseen images across 10 classes.

## Architecture
- Conv2d (3→32 filters) + ReLU + MaxPool
- Conv2d (32→64 filters) + ReLU + MaxPool
- Fully connected: 4096 → 512 → 128 → 10
- Trained with Adam optimizer, CrossEntropyLoss, 10 epochs

## Dataset
CIFAR-10 — 60,000 images across 10 classes: airplane, car, bird, cat, deer, dog, frog, horse, ship, truck.
- 50,000 training images
- 10,000 test images

## Training Environment
- PyTorch 2.11 on Apple M1 (MPS backend)
- Batch size: 64
- Learning rate: 0.001
- Epochs: 10 — loss reduced from 1.35 → 0.08

## How to run
```bash
pip install torch torchvision
python3 cifar_cnn.py
```

## Files
- `cifar_cnn.py` — model definition, training loop, evaluation
- `day1.py` — PyTorch fundamentals (tensors, autograd, MPS)
