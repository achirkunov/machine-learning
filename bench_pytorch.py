# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision"]
# ///
"""PyTorch benchmark: same model as Rust implementation.
Model: linear(784->10) + bias -> softmax -> cross_entropy
Hyperparams: lr=0.03, batch_size=250, epochs=10, SGD
"""

import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load MNIST
    transform = transforms.ToTensor()  # [0,1] floats, same as Rust
    train_dataset = datasets.MNIST("./mnist_data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./mnist_data", train=False, download=True, transform=transform)

    batch_size = 250
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Single linear layer (784 -> 10), same as Rust matmul + bias
    model = nn.Linear(784, 10).to(device)

    # CrossEntropyLoss = log_softmax + nll_loss (matches Rust softmax + cross_entropy)
    # reduction='mean' with lr=0.03 matches Rust's sum loss with lr=0.03/batch_size
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)

    print(f"Training for 10 epochs with lr=0.03, batch_size={batch_size}")
    print("-------------------------------------------")

    start = time.time()

    for epoch in range(10):
        total_loss = 0.0
        samples = 0

        for images, labels in train_loader:
            images = images.view(-1, 784).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            samples += images.size(0)

        # Test accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(-1, 784).to(device)
                labels = labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / samples
        test_acc = correct / total
        print(f"Epoch {epoch+1} complete - avg_loss: {avg_loss:.4f}, test_acc: {test_acc*100:.2f}%")
        print("-------------------------------------------")

    elapsed = time.time() - start

    # Final evaluation
    correct_train = 0
    total_train = 0
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.view(-1, 784).to(device)
            labels = labels.to(device)
            preds = model(images).argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

    print(f"\nFinal Results:")
    print(f"  Train accuracy: {correct_train/total_train*100:.2f}%")
    print(f"  Test accuracy:  {correct/total*100:.2f}%")
    print(f"  Training time:  {elapsed:.2f}s")

if __name__ == "__main__":
    main()
