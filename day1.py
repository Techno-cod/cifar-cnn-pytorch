import torch

# Check device
device = torch.device("mps")
print(f"Using device: {device}")

# Create tensors
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# Basic ops
print("\nMatrix A:\n", a)
print("Matrix B:\n", b)
print("A + B:\n", a + b)
print("A x B (matmul):\n", torch.matmul(a, b))

# Move to M1 GPU
a_mps = a.to(device)
b_mps = b.to(device)
result = torch.matmul(a_mps, b_mps)
print("\nMatmul on MPS (M1 GPU):\n", result)

# Gradients — the core of deep learning
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1   # y = x² + 2x + 1
y.backward()              # compute dy/dx
print(f"\nx = {x.item()}")
print(f"y = x² + 2x + 1 = {y.item()}")
print(f"dy/dx at x=3 → expected 8, got: {x.grad.item()}")

# How a neural network layer works
import torch.nn as nn

layer = nn.Linear(3, 1)  # 3 inputs → 1 output
sample_input = torch.tensor([[1.0, 2.0, 3.0]])
output = layer(sample_input)

print("\n--- Neural network layer ---")
print("Input:", sample_input)
print("Output:", output)
print("Weight:", layer.weight)
print("Bias:", layer.bias)