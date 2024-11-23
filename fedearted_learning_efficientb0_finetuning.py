import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from torch.utils.data import WeightedRandomSampler
import os

def create_weighted_sampler(dataset):
    targets = [label for _, label in dataset.samples]
    class_counts = np.bincount(targets)
    print(f"Class distribution: {class_counts}")
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(targets),
        y=targets
    )
    weights = torch.tensor(class_weights, dtype=torch.float)
    sample_weights = weights[targets]
    
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = r'D:\DR_bisma\Datasets\DR-APTOS\train'
    valid_dir = r'D:\DR_bisma\Datasets\DR-APTOS\valid'
    test_dir = r'D:\DR_bisma\Datasets\DR-APTOS\test'
    
    for dir_path in [train_dir, valid_dir, test_dir]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")

    try:
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        valid_dataset = datasets.ImageFolder(root=valid_dir, transform=valid_transform)
        test_dataset = datasets.ImageFolder(root=test_dir, transform=valid_transform)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    sampler = create_weighted_sampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model = models.efficientnet_b0(pretrained=True)
    
    for name, param in model.named_parameters():
        if "features.7" not in name and "features.8" not in name:
            param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 2)
    )

    model = model.to(device)

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to(device))
    
    optimizer = optim.AdamW([
        {'params': (p for n, p in model.named_parameters() if "features" in n and p.requires_grad), 
         'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=0.01)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    num_epochs = 50
    best_val_acc = 0
    patience = 10
    epochs_without_improvement = 0

    try:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                train_loss += loss.item()

            train_acc = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)

            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    val_loss += loss.item()

            val_acc = 100 * val_correct / val_total
            avg_val_loss = val_loss / len(valid_loader)
            
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                }, 'mobilenet_weights_aptos.pth')
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print("Early stopping triggered")
                    break

        # Test evaluation with error handling
        checkpoint = torch.load('mobilenet_weights_aptos.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        test_correct = 0
        test_total = 0
        class_correct = [0] * 2
        class_total = [0] * 2

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                # Per-class accuracy with error handling
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred = predicted[i].item()
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1

        test_accuracy = 100 * test_correct / test_total
        print(f'Final Test Accuracy: {test_accuracy:.2f}%')
        
        # Print per-class accuracy with error handling
        for i in range(2):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                print(f'Accuracy of class {i}: {class_acc:.2f}%')
            else:
                print(f'No samples for class {i} in test set')

    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise

if __name__ == "__main__":
    train_model()