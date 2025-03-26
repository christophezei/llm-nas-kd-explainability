import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, InterpolationMode
import timm
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
import numpy as np 
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import LLMGenModel

# Define helper classes and functions
class CustomDatasetWithTransformations(Dataset):
    def __init__(self, dataset, transform_student, transform_teacher):
        self.dataset = dataset
        self.transform_student = transform_student
        self.transform_teacher = transform_teacher

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        student_image = self.transform_student(image)  # Apply student's transformation
        teacher_image = self.transform_teacher(image)  # Apply teacher's transformation
        return student_image, teacher_image, label
    
# Updated Function to Handle Custom Dataset Structure
def evaluate_model_with_topk(model, data_loader, device, criterion, topk=(1, 5)):
    model.eval()
    running_loss = 0.0
    correct_top1, correct_top5, total = 0, 0, 0

    with torch.no_grad():
        for batch in data_loader:
            # Adjust based on whether teacher images are provided
            if len(batch) == 3:
                images, _, labels = batch  # Ignore teacher images for evaluation
            else:
                images, labels = batch
            
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            # Top-1 Accuracy
            _, top1_predicted = outputs.topk(1, dim=1)
            correct_top1 += top1_predicted.eq(labels.view(-1, 1)).sum().item()

            # Top-5 Accuracy
            _, top5_predicted = outputs.topk(5, dim=1)
            correct_top5 += top5_predicted.eq(labels.view(-1, 1).expand_as(top5_predicted)).sum().item()

    avg_loss = running_loss / total
    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total
    return avg_loss, top1_acc, top5_acc

def fine_tune_with_distinct_preprocessing(
    student_model, teacher_model, train_loader, val_loader, test_loader,
    device, num_epochs=10, lr=0.01, weight_decay=1e-5
):

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    distillation_criterion = nn.KLDivLoss(reduction="batchmean")
    alpha = 0.4
    final_alpha = 0.8
    initial_temp = 5
    final_temp = 3
    temperature = 10

    student_model.to(device)
    teacher_model.to(device)
    teacher_model.eval()
    # Define the optimizer
    warmup_epochs = 10
    optimizer = optim.SGD(student_model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader),
        epochs=num_epochs, pct_start=0.3
    )
    
    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        correct, total = 0, 0
        alpha = alpha = alpha + (final_alpha - alpha) * (epoch / num_epochs)
 
        for student_images, teacher_images, labels in train_loader:
            student_images, teacher_images, labels = (
                student_images.to(device),
                teacher_images.to(device),
                labels.to(device),
            )

            optimizer.zero_grad()

            # Forward pass
            student_outputs = student_model(student_images)
            with torch.no_grad():
                teacher_outputs = teacher_model(teacher_images)

            # Compute losses
            ce_loss = criterion(student_outputs, labels)

            distillation_loss = distillation_criterion(
                F.log_softmax(student_outputs / temperature, dim=1),
                F.softmax(teacher_outputs / temperature, dim=1),
            ) * (temperature ** 2)

            # Combine losses using alpha
            loss = alpha * ce_loss + (1 - alpha) * distillation_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update metrics
            running_loss += loss.item() * student_images.size(0)
            _, predicted = torch.max(student_outputs, 1)
            total += labels.size(0)
        
            correct += (predicted == labels).sum().item()


        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        scheduler.step()

        # Validation evaluation
        val_loss, val_top1, val_top5 = evaluate_model_with_topk(student_model, val_loader, device, criterion)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Top-1 Acc: {val_top1:.2f}%, Val Top-5 Acc: {val_top5:.2f}%")

    # Test evaluation
    test_loss, test_top1, test_top5 = evaluate_model_with_topk(student_model, test_loader, device, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Top-1 Accuracy: {test_top1:.2f}%, Test Top-5 Accuracy: {test_top5:.2f}%")

def main():
    # Load teacher and student models
    teacher_model = timm.create_model("timm/vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=False)
    teacher_model.head = nn.Linear(teacher_model.head.in_features, 100)
    teacher_model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            "https://huggingface.co/edadaltocg/vit_base_patch16_224_in21k_ft_cifar100/resolve/main/pytorch_model.bin",
            map_location="cuda",
            file_name="vit_base_patch16_224_in21k_ft_cifar100.pth",
        )
    )
    print("Teacher model loaded successfully.")

    weights_path = "./models/Llama8b/LMaNet_elite.pth"
    with open("./models/Llama8b/LMaNet_elite.json", "r") as f:
        best_config = json.load(f)["candidate_config"]

    student_model = LLMGenModel(
        config=best_config, 
        num_classes=100, 
        dropout_rate=0.05,
        pool_type="average",
    )

    print(best_config)
    print(student_model)
    state_dict = torch.load(weights_path, map_location="cuda")
    student_model.load_state_dict(state_dict)
    print("Student model loaded successfully.")

    input_size = 160
    teacher_input_size = 224
    num_classes = 100

    
    transform_student_train = transforms.Compose([
        transforms.Resize(input_size, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    ])

    transform_teacher_train = transforms.Compose([
        transforms.Resize(teacher_input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform_val_test = transforms.Compose([
        transforms.Resize(input_size, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    ])

    # Load CIFAR-100 dataset
    cifar100_train = datasets.CIFAR100(root="./data", train=True, download=True)

    # Perform stratified split for validation
    train_labels = np.array(cifar100_train.targets)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, val_indices = next(splitter.split(np.zeros(len(train_labels)), train_labels))

    train_subset = Subset(cifar100_train, train_indices)
    val_subset = Subset(cifar100_train, val_indices)

    train_dataset = CustomDatasetWithTransformations(train_subset, transform_student_train, transform_teacher_train)
    val_dataset = CustomDatasetWithTransformations(val_subset, transform_val_test, transform_val_test)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)

    test_loader = DataLoader(datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_val_test),
                             batch_size=256, shuffle=False, num_workers=2)

    fine_tune_with_distinct_preprocessing(
        student_model, teacher_model, train_loader, val_loader, test_loader,
        device="cuda", num_epochs=50, lr=2e-3, weight_decay=1e-4
    )
    # Save the fine-tuned student model
    torch.save(student_model.state_dict(),"./models/Llama8b/vit_LMaNet_elite.pth")

if __name__ == "__main__":
    main()
