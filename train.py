import torch
import torch.nn as nn
import numpy as np
import math
import os
import json
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import precision_score, recall_score, f1_score
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, InterpolationMode
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, random_split
from helper import load_best_candidate, compute_model_stats, compute_peak_sram
from model import LLMGenModel
from PIL import Image
from torch.utils.data import Dataset

def load_cifar100(batch_size=64, image_size=160):
    """
    Load CIFAR-100 dataset with training, validation, and testing splits, resizing images to 160x160.

    Args:
        batch_size (int): Batch size for DataLoaders.
        image_size (int): Target image size for resizing.

    Returns:
        train_loader, val_loader, test_loader: Data loaders for CIFAR-100.
    """
    transform_train = transforms.Compose([
    transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
    AutoAugment(policy=AutoAugmentPolicy.CIFAR10, interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    ])

    # Load datasets
    dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

    # Split train and validation sets
    val_size = int(0.05 * len(dataset))  # 5% validation set
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Update validation dataset transformation to match test transform
    val_dataset.dataset.transform = transform_test

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

def lr_lambda(epoch, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        # Linear warmup from 0 to target LR
        return epoch / warmup_epochs
    else:
        # Cosine decay after warmup
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

def mixup_data_with_spatial(x, y, alpha=0.2, spatial_transform=None):
    """
    Apply MixUp with optional spatial transformations.

    Args:
        x (Tensor): Batch of input images.
        y (Tensor): Batch of labels.
        alpha (float): Parameter for Beta distribution.
        spatial_transform (callable): Spatial transformation function.
    
    Returns:
        Tensor: Mixed images.
        Tensor: Labels for the first image in the mix.
        Tensor: Labels for the second image in the mix.
        float: Mixup coefficient (lambda).
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # Apply spatial transformation if provided
    if spatial_transform is not None:
        x_transformed = torch.stack([spatial_transform(img) for img in x])
        x_index_transformed = torch.stack([spatial_transform(x[idx]) for idx in index])
    else:
        x_transformed = x
        x_index_transformed = x[index, :]

    # MixUp images and labels
    mixed_x = lam * x_transformed + (1 - lam) * x_index_transformed
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def evaluate_model(
    model, train_loader, val_loader, test_loader, device="cuda", num_epochs=120, 
    optimizer_type='SGD', lr=0.1, weight_decay=1e-5, is_mix_up=False,
):

    model.to(device)
    # Initialize optimizer and scheduler
    if optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif optimizer_type == 'SGD':
        warmup_epochs = 20
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: lr_lambda(epoch, warmup_epochs, num_epochs))
        spatial_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        ])
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler()
    print(f"Full Training Phase:")
    
    val_acc_history, val_f1_history = [], []
    best_val_acc = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}: LR = {scheduler.get_last_lr()[0]}")
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            # Apply MixUp with spatial transformations
            if is_mix_up:
                images, labels_a, labels_b, lam = mixup_data_with_spatial(images, labels, alpha=0.2, spatial_transform=spatial_transform)
                images, labels_a, labels_b = images.to(device), labels_a.to(device), labels_b.to(device)
            else:
                images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                if is_mix_up:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Track loss
            running_loss += loss.item() * images.size(0)

            # MixUp accuracy (soft approximation)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            if is_mix_up:
                correct += lam * (predicted == labels_a).sum().item() + (1 - lam) * (predicted == labels_b).sum().item()
            else:
                correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = (correct / total) * 100
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= val_total
        val_acc = (val_correct / val_total) * 100
        val_acc_history.append(val_acc)
        val_precision = precision_score(all_labels, all_preds, average='macro')
        val_recall = recall_score(all_labels, all_preds, average='macro')
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_f1_history.append(val_f1)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.2f}")

    test_metrics = {}

    print("Testing model...")
    model.eval()
    test_correct, test_total, top5_correct = 0, 0, 0
    all_test_preds, all_test_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
            _, top1_predicted = torch.max(outputs, 1)
            top5_predicted = torch.topk(outputs, 5, dim=1).indices

            test_total += labels.size(0)
            test_correct += (top1_predicted == labels).sum().item()
            top5_correct += torch.sum(top5_predicted.eq(labels.unsqueeze(1))).item()
            all_test_preds.extend(top1_predicted.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())

    test_acc = (test_correct / test_total) * 100
    test_top5_acc = (top5_correct / test_total) * 100
    test_precision = precision_score(all_test_labels, all_test_preds, average='macro')
    test_recall = recall_score(all_test_labels, all_test_preds, average='macro')
    test_f1 = f1_score(all_test_labels, all_test_preds, average='macro')
    print(f"Test - Acc: {test_acc:.2f}%, Top-5 Acc: {test_top5_acc:.2f}%, F1: {test_f1:.2f}")
    test_metrics = {
        'test_acc': test_acc,
        'test_top5_acc': test_top5_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1
    }

    # Compute Model Stats
    model.cpu()
    stats = compute_model_stats(model, input_shape=(3, 160, 160))
    print(f"Model Stats: Params: {stats['num_params_millions']:.2f}M, Size: {stats['model_size_MB']:.2f}MB, MACs: {stats['macs_millions']:.2f}M")

    return {
        'train_loss': train_loss,
        'val_acc': best_val_acc,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1,
        **test_metrics,
        'num_params_millions': stats['num_params_millions'],
        'model_size_MB': stats['model_size_MB'],
        'macs_millions': stats['macs_millions']
    }
    
def main():
    saved_config_path = "models/Llama8b/LMaNet_elite.json"
    weights_path = "models/Llama8b/LMaNet_elite.pth"

    # Load the best candidate configuration and metrics
    best_config, metrics = load_best_candidate(filepath=saved_config_path)

    # Build the model from the loaded configuration
    model = LLMGenModel(
        config=best_config, 
        num_classes=100, 
        dropout_rate=0.05,
        pool_type="average", 
    )
    print(model)

    num_epochs = 120
    optimizer_type = "SGD"
    weight_decay = 1e-4
    lr=0.3
    sram = compute_peak_sram(model, (1, 3, 160, 160), dtype=torch.int8)
    stats = compute_model_stats(model, input_shape=(3, 160, 160))
    print(f"Model Stats: Params: {stats['num_params_millions']:.2f}M, Size: {stats['model_size_MB']:.2f}MB, MACs: {stats['macs_millions']:.2f}M")
    print("Peak sram: ", sram)
    train_loader, val_loader, test_loader = load_cifar100(batch_size=128, image_size=160)

    extended_metrics = evaluate_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_epochs=num_epochs,
        optimizer_type=optimizer_type, 
        lr=lr, 
        weight_decay=weight_decay,
        is_mix_up=True,
    )
    # Save the model
    torch.save(model.state_dict(), weights_path)
    print("\nExtended Training Metrics:")
    for key, value in extended_metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()