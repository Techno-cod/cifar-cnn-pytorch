import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models

TRAIN = False  # set to True to retrain

device = torch.device("mps")
print(f"Training on: {device}")

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=False, transform=train_transform)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=False, transform=test_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=64, shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

if TRAIN:
    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': [p for n, p in model.named_parameters() if 'fc' not in n], 'lr': 0.0001},
        {'params': model.fc.parameters(), 'lr': 0.001}
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    print("\nStarting full fine-tuning...")
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/10 — loss: {running_loss/len(trainloader):.3f} — lr: {current_lr:.6f}")

    torch.save(model.state_dict(), 'resnet18_cifar10.pth')
    print("Model saved.")

# Evaluation
model.load_state_dict(torch.load('resnet18_cifar10.pth', map_location=device))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nTest accuracy: {accuracy:.2f}%")
print(f"Correct: {correct}/{total}")


