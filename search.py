import os
import json
import random
import re
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, SequentialLR, LinearLR, LambdaLR
from torch.utils.data import DataLoader, random_split

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, InterpolationMode

from ptflops import get_model_complexity_info

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from model import LLMGenModel
from helper import  compute_model_stats, compute_peak_sram
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.onnx
# Ensure we don't run into parallelism issues with tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ===============================
#           Models
# ===============================

import torch
import torch.nn as nn
from ptflops import get_model_complexity_info

# ===============================
#       Data Loading
# ===============================

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


# ===============================
#        Search Space
# ===============================

def hierarchical_search_space():
    """
    Defines the refined search space for hierarchical NAS.

    Returns:
        dict: Hierarchical search space parameters.
    """
    return {
        'output_channels': [16, 24, 32, 48, 64, 96, 128, 160],
        'num_layers': [1, 2, 3, 4, 6],  # Number of layers in a block Ni 
        'kernel_size': [3, 5, 7],  # Kernel size per layer (added 7 for more diversity) [1, 3, 5, 7]
        'stride': [1, 2],  # Stride for downsampling
        'expansion_factor': [3, 4, 6],
        'use_se': [True, False],  # Squeeze-and-Excitation
        'se_ratio': [0.25, 0.5],  # Squeeze-and-Excitation ratio
        'conv_type': ['depthwise', 'mbconv'],  # ConvOp
        'skip_op': ['identity', 'residual'],  # SkipOpœ
        'activation': ['relu6', 'leakyrelu', 'swish'],  # Activation functions
    }

# ===============================
#          LLM Setup
# ===============================

def initialize_llm():
    """
    Initializes the Large Language Model (LLM) for architecture generation.

    Returns:
        model, tokenizer: The initialized LLM model and tokenizer.
    """
    if is_bfloat16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # model_name = "unsloth/Qwen2.5-3B-Instruct"
    # model_name = "unsloth/Llama-3.2-3B-Instruct" 
    # model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=16384,
        load_in_4bit=False,
        # trust_remote_code=True
    )

    FastLanguageModel.for_inference(model)

    return model, tokenizer


def interact_with_llm_chat(prompt, model, tokenizer, device=None, max_new_tokens=2048, temperature=1.5, min_p=0.1, max_retries=3):
    """
    Interacts with the LLM in a chat-like format and handles response validation and retries.

    Args:
        prompt (str): The prompt to be sent to the LLM.
        model: Pretrained LLM model.
        tokenizer: Tokenizer for the model.
        device (torch.device, optional): The device to use (CPU or GPU). Default is auto-detected.
        max_new_tokens (int): Maximum number of tokens to generate. Default is 1024.
        temperature (float): Sampling temperature. Default is 1.5.
        min_p (float): Minimum nucleus sampling probability. Default is 0.1.
        max_retries (int): Maximum number of retries for invalid responses. Default is 3.

    Returns:
        str: The cleaned response text from the LLM.
    """
    # Detect device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for attempt in range(max_retries):
        try:
            # Prepare the inputs using the chat template
            inputs = tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
            
            # Get the index for generation
            gen_idx = len(inputs[0])

            # Generate response
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                min_p=min_p
            )

            # Decode the generated response
            response_text = tokenizer.batch_decode(outputs[:, gen_idx:], skip_special_tokens=True)[0]

            # Debug: Print raw response
            print(f"Raw LLM Response: {response_text}")

            # Validate response and return if valid
            if response_text.strip():
                return response_text.strip()

        except Exception as e:
            print(f"Error during LLM interaction on attempt {attempt + 1}/{max_retries}: {e}")

    # Return empty string if all retries fail
    print("Failed to generate a valid response after retries.")
    return ""

# ===============================
#      Prompt Generation
# ===============================

def generate_feedback_prompt(
    in_channels, search_space, num_blocks, answer_exp, feedback=None, 
    input_image_size=(160, 160), mac_limit=350, iter_numb=0
):
    feedback_section = f"Feedback for improvement:\n{feedback}\n" if feedback else ""
    prompt = f"""
    You are a neural architecture design algorithm that exclusively outputs configurations in JSON format.
    Your task is to generate a lightweight neural network architecture tailored for the CIFAR-100 dataset, specifically optimized for image classification.
    The current state-of-the-art model for this task achieves an accuracy of 75% with a computational cost of 259 million MACs.
    Your objective is to design a model that achieves comparable or superior results, targeting an accuracy of at least 75%.

    Constraints:
    - Minimize RAM usage during inference by reducing intermediate activation size.
    - Prioritize stride=2 in early blocks to downsample spatial dimensions.
    - Use smaller `expansion_factor` and `output_channels` in early blocks to reduce feature map size.
    - Gradually increase the `output_channels` in later blocks.
    - Limit the use of SE blocks and their ratio to avoid high activation memory.
    - The image size is {input_image_size[0]}x{input_image_size[1]}.
    - The kernel size and stride must ensure valid output dimensions ({input_image_size[0]} >= 0).
    - Ensure the total MACs for the network does not exceed {mac_limit}M.
    - Only use values from the hierarchical_search_space:
    {json.dumps(search_space, indent=4)}

    Example instruction: "Generate a lightweight model for image classification using {num_blocks} blocks."
    Example answer:"
    {json.dumps(answer_exp, indent=4)}

    Now, here is my instruction: Generate a lightweight model for image classification using {num_blocks} blocks.
    Use hierarchical_search_space to find the best config.
    {feedback_section}
    Please do not include anything else other than the JSON in your response.
    """
    return prompt

# ===============================
#      Response Parsing
# ===============================

def extract_config_from_llm_response(response):
    """
    Extracts a JSON configuration from the LLM response.

    Args:
        response (str): The full LLM response as a string.

    Returns:
        list: Parsed block configurations as a list of dictionaries.
    """
    try:
        # Identify where the JSON content starts and ends
        config_start = response.find("[")  # Start of the JSON list
        config_end = response.rfind("]")  # End of the JSON list

        if config_start == -1 or config_end == -1:
            raise ValueError("No valid JSON array found in the response.")

        # Extract the JSON substring
        json_content = response[config_start:config_end + 1]

        # Parse the JSON string into a Python list of dictionaries
        config = json.loads(json_content)

        # Validate the parsed configuration is a list of dictionaries
        if not isinstance(config, list) or not all(isinstance(block, dict) for block in config):
            raise ValueError("Parsed configuration is not a valid list of dictionaries.")

        return config

    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON: {e}")

    except ValueError as e:
        raise ValueError(f"Validation Error: {e}")


# ===============================
#      MACs Validation
# ===============================
def quick_validate_macs(candidate_config, min_mac, max_mac):
    """
    Quickly validates if the candidate's MACs meet the specified minimum and maximum constraints.

    Args:
        candidate_config (dict): Candidate network configuration.
        min_mac (float): Minimum MAC limit in millions.
        max_mac (float): Maximum MAC limit in millions.

    Returns:
        bool, float: 
            - True and macs_millions if within [min_mac, max_mac].
            - False and macs_millions otherwise.
    """
    try:
        # Build the model from the configuration
        candidate_model = LLMGenModel(candidate_config, num_classes=100, use_final_layer=False)

        # Use ptflops to estimate MACs
        input_res = (3, 160, 160)  # Define the input resolution
        macs, _ = get_model_complexity_info(candidate_model, input_res, as_strings=False, print_per_layer_stat=False)
        macs_millions = macs / 1e6

        # Validate MACs against the min and max limits
        if macs_millions < min_mac:
            print(f"Candidate MACs too low: {macs_millions:.2f}M < {min_mac:.2f}M")
            return False, macs_millions
        elif macs_millions > max_mac:
            print(f"Candidate exceeds MAC limit: {macs_millions:.2f}M > {max_mac:.2f}M")
            return False, macs_millions
        else:
            print(f"Candidate MACs: {macs_millions:.2f}M (within range: {min_mac:.2f}M - {max_mac:.2f}M)")
            return True, macs_millions

    except Exception as e:
        print(f"Error estimating MACs: {e}")
        return False, float('inf')

def lr_lambda(epoch, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        # Linear warmup from 0 to target LR
        return epoch / warmup_epochs
    else:
        # Cosine decay after warmup
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

# ===============================
#      Evaluation Function
# ===============================
def evaluate_candidate(
    model, train_loader, val_loader, test_loader, device="cuda", num_epochs=40, 
    optimizer_type='SGD', lr=0.1, weight_decay=1e-5, mac_limit=350, 
    baseline_acc=70, evaluation_phase='initial'
):
    """
    Evaluates a candidate model for accuracy, precision, recall, F1-score, MACs, and size.

    Args:
        model (nn.Module): Candidate model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for testing data.
        device (str or torch.device): Device for evaluation.
        num_epochs (int): Number of epochs for training.
        optimizer_type (str): Type of optimizer ('AdamW' or 'SGD').
        lr (float): Learning rate.
        weight_decay (float): Weight decay for optimizer.
        mac_limit (float): MAC limit in millions.
        baseline_acc (float): Baseline validation accuracy for promising candidates.
        evaluation_phase (str): 'initial' or 'final' to determine training depth.

    Returns:
        dict: Performance metrics including accuracy, precision, recall, F1-score, MACs, and size.
    """
    model.to(device)
    
    # Initialize optimizer and scheduler
    if optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif optimizer_type == 'SGD':
        warmup_epochs = 10
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler()
    print(f"Training Phase: {evaluation_phase}")
    
    val_acc_history, val_f1_history = [], []
    best_val_acc = 0
    patience = num_epochs
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}: LR = {scheduler.get_last_lr()[0]}")
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
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

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Analyze Validation Trends
    improvement_rate = (val_acc_history[-1] - val_acc_history[0]) / len(val_acc_history)
    epochs = np.arange(len(val_acc_history))
    slope, _ = np.polyfit(epochs, val_acc_history, 1)
    
    # Determine if the candidate is promising
    promising = (
        val_acc_history[-1] > baseline_acc or
        val_f1_history[-1] > 0.55 or
        slope > 0.5 or
        improvement_rate > 1.0
    )
    print(f"Promising Candidate: {promising} | Final Val Acc: {val_acc_history[-1]:.2f}%, Slope: {slope:.2f}, Improvement Rate: {improvement_rate:.2f}")

    # Test Promising Candidates
    test_metrics = {}
    if promising:
        print("Testing promising candidate...")
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
        'promising': bool(promising),  # Explicitly convert to Python bool
        'improvement_rate': improvement_rate,
        'slope': slope,
        **test_metrics,
        'num_params_millions': stats['num_params_millions'],
        'model_size_MB': stats['model_size_MB'],
        'macs_millions': stats['macs_millions']
    }

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
    model, train_loader, val_loader, test_loader, device="cuda", num_epochs=40, 
    optimizer_type='SGD', lr=0.1, weight_decay=1e-5, mac_limit=400, 
    baseline_acc=70, evaluation_phase='initial', is_mix_up=False, resume_trainig=False
):
    """
    Evaluates a candidate model for accuracy, precision, recall, F1-score, MACs, and size.

    Args:
        model (nn.Module): Candidate model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for testing data.
        device (str or torch.device): Device for evaluation.
        num_epochs (int): Number of epochs for training.
        optimizer_type (str): Type of optimizer ('AdamW' or 'SGD').
        lr (float): Learning rate.
        weight_decay (float): Weight decay for optimizer.
        mac_limit (float): MAC limit in millions.
        baseline_acc (float): Baseline validation accuracy for promising candidates.
        evaluation_phase (str): 'initial' or 'final' to determine training depth.

    Returns:
        dict: Performance metrics including accuracy, precision, recall, F1-score, MACs, and size.
    """
    model.to(device)
    # Initialize optimizer and scheduler
    if optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif optimizer_type == 'SGD':
        if resume_trainig:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
            warmup_epochs = 20
        else:
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
    print(f"Training Phase: {evaluation_phase}")
    
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

    print("Testing candidate...")
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
# ===============================
#      LLM Interaction
# ===============================

def build_candidate_network_with_context_single_proposal(
    llm_model, tokenizer, input_channels, input_size, num_blocks, search_space, answer_exp, iter_numb=0, feedback=None, max_retries=10
):
    """
    Build a candidate network configuration by querying the LLM for the entire network in one go,
    with retries for faulty configurations and explicit feedback for invalid responses.

    Args:
        llm_model: Pretrained LLM model.
        tokenizer: Tokenizer for the LLM model.
        input_channels (int): Starting input channels for the network.
        input_size (tuple): Starting input size as (height, width).
        num_blocks (int): Number of blocks in the candidate network.
        search_space (dict): Search space for block parameters.
        answer_exp (list): Example JSON answer for the instruction.
        feedback (str, optional): Feedback for the network generation.
        max_retries (int): Maximum retries allowed for a faulty configuration.

    Returns:
        list: Fully constructed network configuration or None if retries failed.
    """
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Generate a concise prompt for the entire network
            prompt = generate_feedback_prompt(
                in_channels=input_channels,
                search_space=search_space,
                num_blocks=num_blocks,
                answer_exp=answer_exp,
                iter_numb=iter_numb,
                feedback=feedback,
                input_image_size=input_size,
                mac_limit=350  # Ensure MAC limit is respected
            )
            messages = [{"role": "user", "content": prompt}]
            # Interact with the LLM to generate the entire network
            response = interact_with_llm_chat(messages, llm_model, tokenizer)
           
            # Extract the network configuration from the response
            network_config = extract_config_from_llm_response(response)
            print(f"Generated Network Configuration: {network_config}")

            return network_config  # Return the successfully generated network

        except ValueError as e:
            retry_count += 1
            print(f"Error in generating network configuration: {e}")
            
            # Provide feedback to the LLM with an example if format is not respected
            if retry_count < max_retries:
                feedback = (
                    "The provided network configuration was not valid. "
                    "Ensure the format matches the following example:\n\n"
                    f"{json.dumps(answer_exp, indent=4)}\n\n"
                    "Please regenerate the network with valid configurations."
                )
                print(f"Retrying network generation ({retry_count}/{max_retries}) with updated feedback...")
            else:
                print("Maximum retries reached. Returning failure feedback.")

    print(f"Failed to generate a valid network configuration after {max_retries} retries.")
    return None  # Return None if retries failed

# ===============================
#      Pareto Set Management
# ===============================

def dominates(metrics_a, metrics_b):
    """
    Determines if metrics_a Pareto dominates metrics_b.

    Args:
        metrics_a (dict): Metrics of candidate A.
        metrics_b (dict): Metrics of candidate B.

    Returns:
        bool: True if A dominates B, False otherwise.
    """
    # Ensure all required metrics are present
    required_metrics = ['test_acc', 'num_params_millions', 'macs_millions']
    if not all(metric in metrics_a for metric in required_metrics) or not all(metric in metrics_b for metric in required_metrics):
        raise ValueError("Metrics must include 'test_acc', 'num_params_millions', and 'macs_millions'.")

    # Check dominance conditions
    better_or_equal = (
        metrics_a['test_acc'] >= metrics_b['test_acc'] and
        metrics_a['num_params_millions'] <= metrics_b['num_params_millions'] and
        metrics_a['macs_millions'] <= metrics_b['macs_millions']
    )
    strictly_better = (
        metrics_a['test_acc'] > metrics_b['test_acc'] or
        metrics_a['num_params_millions'] < metrics_b['num_params_millions'] or
        metrics_a['macs_millions'] < metrics_b['macs_millions']
    )
    return better_or_equal and strictly_better

def update_pareto_set(candidates):
    """
    Computes the Pareto front from a list of candidates.

    Args:
        candidates (list): List of candidates with 'metrics'.

    Returns:
        list: Pareto front candidates.
    """
    pareto_front = []
    for candidate in candidates:
        dominated = any(dominates(existing['metrics'], candidate['metrics']) for existing in pareto_front)
        if not dominated:
            pareto_front = [existing for existing in pareto_front if not dominates(candidate['metrics'], existing['metrics'])]
            pareto_front.append(candidate)
    return pareto_front


def generate_pareto_feedback(pareto_set):
    """
    Generates feedback for improving future candidates based on the Pareto set, including architecture summaries.

    Args:
        pareto_set (list): Current Pareto-optimal set.

    Returns:
        str: Feedback for optimizing future candidates.
    """
    if not pareto_set:
        return "No Pareto-optimal candidates yet. Focus on improving accuracy, reducing MACs, and minimizing parameters."

    feedback = "Optimize future candidates to balance accuracy, MACs, and parameters.\n"
    feedback += "Current Pareto-optimal candidates:\n"

    for idx, candidate in enumerate(pareto_set):
        metrics = candidate['metrics']
        config = candidate['candidate_config']
        feedback += (
            f"Candidate {idx + 1}:\n"
            f" Accuracy: {metrics['test_acc']:.2f}%, Params: {metrics['num_params_millions']:.2f}M, MACs: {metrics['macs_millions']:.2f}M\n"
        )
        try:
            # Convert blocks to JSON format for cleaner output
            architecture_json = json.dumps(config['blocks'], indent=4)
            feedback += f"  Architecture:\n{architecture_json}\n\n"
        except KeyError as e:
            error = f"  Error retrieving architecture: {str(e)}\n\n"

    # Summarize trends
    avg_accuracy = sum(c['metrics']['test_acc'] for c in pareto_set) / len(pareto_set)
    avg_macs = sum(c['metrics']['macs_millions'] for c in pareto_set) / len(pareto_set)
    avg_params = sum(c['metrics']['num_params_millions'] for c in pareto_set) / len(pareto_set)

    feedback += f"Summary:\nAverage Accuracy: {avg_accuracy:.2f}%\n"
    feedback += f"Average MACs: {avg_macs:.2f}M\n"
    feedback += f"Average Params: {avg_params:.2f}M\n"

    # Highlight potential areas for improvement
    if avg_accuracy > 70:
        feedback += "Focus on reducing MACs or parameters without compromising accuracy.\n"
    elif avg_macs > 300:
        feedback += "Focus on reducing MACs below 300M.\n"
    else:
        feedback += "Explore more configurations with higher accuracy while maintaining low MACs and parameters.\n"

    return feedback


# ===============================
#      Feedback Loop
# ===============================

def feedback_loop_single_proposal(
    llm_model,
    tokenizer,
    search_space,
    train_loader,
    val_loader,
    test_loader,
    answer_exp,
    initial_in_channels=3,
    num_blocks=5,
    max_iterations=100,
    experiment_name="Llma8b",
    mac_limit=400,
):
    """
    Feedback loop to iteratively generate and evaluate a single proposed network from the LLM
    with Pareto-optimal selection and dynamic feedback.

    Args:
        llm_model: The LLM model instance.
        tokenizer: The tokenizer for the LLM model.
        search_space (dict): The search space for generating network configurations.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        test_loader: DataLoader for testing data.
        answer_exp (list): Example JSON answer for the instruction.
        initial_in_channels (int): The number of input channels for the network.
        num_blocks (int): Number of blocks in the network.
        max_iterations (int): Maximum iterations to refine candidates.
        experiment_name (str): Name for saving the best candidate and experiment results.
        mac_limit (int): Maximum allowable MACs for the network.

    Returns:
        dict: Best network configuration and its performance metrics.
        list: Pareto-optimal set of candidates.
        list: History of all evaluated candidates and their metrics.
    """
    best_candidate = None
    best_metrics = {'test_acc': 0, 'num_params_millions': float('inf'), 'macs_millions': float('inf')}
    feedback = None

    # Initialize history, Pareto-optimal set
    history = []
    pareto_set = []
    candidate_config_history = []
    all_candidates_path = f"{experiment_name}/all_trained_candidates.json"
    
    # Initialize the JSON file if it doesn't exist
    if not os.path.exists(all_candidates_path):
        with open(all_candidates_path, "w") as f:
            json.dump([], f)

    for iteration in range(max_iterations):
        print(f"\nIteration {iteration}: Generating Candidate...")

        # Generate candidate network
        blocks = build_candidate_network_with_context_single_proposal(
            llm_model=llm_model,
            tokenizer=tokenizer,
            input_channels=initial_in_channels,
            input_size=(160, 160),
            num_blocks=num_blocks,
            search_space=search_space,
            answer_exp=answer_exp,
            feedback=feedback,
            iter_numb=iteration
        )

        if blocks is None:
            print("Failed to generate a valid candidate. Skipping iteration...")
            continue
        try:
            candidate_config = {
                "input_channels": initial_in_channels,
                "blocks": blocks,
                "pool_type": "average"  # Default pooling type; can be adjusted based on feedback
            }
        except ValueError as e:
            print(f"Error parsing candidate response: {e}")
            continue
        
        # Check if the candidate already exists in history
        existing_candidate = next((cand for cand in candidate_config_history if cand['candidate_config'] == candidate_config), None)

        if existing_candidate:
            # If the candidate configuration is found, retrieve its 'can_config'
            can_config_value = existing_candidate['candidate_config']
            
            # Provide feedback indicating that the configuration has already been evaluated
            feedback = f"This candidate configuration has already been evaluated. Propose a new candidate with unique settings. Previous candidate 'can_config' value: {can_config_value}"
            
            print("Feedback for LLM:", feedback)
            continue
        else:
            # If the candidate configuration doesn't exist in history, create a new entry
            can_config = {'candidate_config': candidate_config}
            candidate_config_history.append(can_config)

        min_mac = 70
        valid_macs, estimated_macs = quick_validate_macs(candidate_config, min_mac=min_mac, max_mac=mac_limit)
        if not valid_macs:
            if estimated_macs < min_mac:
                feedback = f"Increase MACs to at least {min_mac}M. Current estimate is {estimated_macs:.2f}M."
            else:
                feedback = f"Reduce MACs below {mac_limit}M. Current estimate is {estimated_macs:.2f}M."
            print("Feedback for LLM:", feedback)
            continue
        
        # Build and evaluate the candidate model
        candidate_model = LLMGenModel(candidate_config, num_classes=100, dropout_rate=0.2, pool_type=candidate_config.get('pool_type', 'average'))

        # Compute Peak SRAM usage
        sram = compute_peak_sram(candidate_model, (1, 3, 160, 160), dtype=torch.int8)

        max_sram_limit = 0.4
        # Validate Peak SRAM usage
        if sram > max_sram_limit:
            feedback = (
                f"Reduce peak SRAM usage below {max_sram_limit:.2f} MB. "
                f"Current peak SRAM usage is {sram:.2f} MB. "
                "Consider reducing output channels, intermediate activations, or simplifying the model architecture."
            )
            print("Feedback for LLM:", feedback)
            print(f"Current peak SRAM usage is {sram:.2f} MB. ")
            continue
        
        # Query the LLM for an explanation of the design
        explanation_query = (
            "Explain why this design was chosen for the current iteration. "
            "Highlight the reasoning behind the selected architecture parameters, "
            "such as kernel sizes, expansion factors, or activation functions."
        )
        explanation = query_llm_for_explanation(llm_model, tokenizer, candidate_config, explanation_query)
        print(f"LLM Explanation for Iteration {iteration}:\n{explanation}\n")
        metrics = evaluate_candidate(
            candidate_model, train_loader, val_loader, test_loader,
            device="cuda",
            num_epochs=30,
            optimizer_type='SGD',
            lr=0.5,
            weight_decay=1e-4,
            mac_limit=mac_limit,
            evaluation_phase='mini'
        )
        print(f"Candidate Metrics: {metrics}")

        # Save the candidate and its metrics to the history
        candidate = {'candidate_config': candidate_config, 'metrics': metrics}
        # Load the existing candidates from the JSON file if it exists
        with open(all_candidates_path, "r") as f:
            all_candidates = json.load(f)

        # Append the current candidate to the list
        all_candidates.append(candidate)

        # Save the updated list back to the file
        with open(all_candidates_path, "w") as f:
            json.dump(all_candidates, f, indent=4)

        print(f"Candidate added to {all_candidates_path}")

        history.append(candidate)

        # Update the Pareto set
        pareto_set = update_pareto_set(all_candidates)

        # Check if this candidate is the best so far
        if metrics['promising']:
            if metrics['test_acc'] > best_metrics['test_acc']:
                best_candidate = candidate_config
                best_metrics = metrics

                # Save the best candidate's configuration and metrics
                save_candidate(best_candidate, best_metrics, filename=f"{experiment_name}_best_candidate.json")

                # Save the model's weights
                torch.save(candidate_model.state_dict(), f"saved_candidates/feed/{experiment_name}_best_candidate.pth")

                print(f"New best candidate saved with test accuracy: {metrics['test_acc']:.2f}%")
                
            # Generate feedback for the next iteration based on the Pareto set
            # Include the best candidate's architecture and accuracy in the feedback
            best_model_feedback = (
                f"Best Model so far:\n"
                f"Architecture: {best_candidate}\n"
                f"Accuracy: {best_metrics['test_acc']:.2f}%\n\n"
            )
            feedback = best_model_feedback + generate_pareto_feedback(pareto_set)

            print(f"Feedback prompt: {feedback}\n")
            # Save the Pareto set
            os.makedirs("saved_candidates/feed", exist_ok=True)
            with open(f"saved_candidates/feed/{experiment_name}_pareto_set.json", "w") as f:
                json.dump(pareto_set, f, indent=4)
        else:
            # Provide specific feedback on why the candidate was not promising
            feedback = "Architecture not promising due to:\n"
            baseline_acc = 70
            # Check specific reasons why the candidate failed the "promising" condition
            if metrics['val_acc'] <= baseline_acc:
                feedback += f"  - Validation Accuracy ({metrics['val_acc']:.2f}%) did not meet the baseline of {baseline_acc}%.\n"
            if metrics['val_f1'] <= 0.55:
                feedback += f"  - Validation F1 Score ({metrics['val_f1']:.2f}) did not exceed 0.55.\n"

            # Add general advice for improvement
            feedback += "Focus on improving accuracy, F1 score, and validation trends to make the architecture promising.\n"
    return best_candidate, pareto_set, history

def query_llm_for_explanation(llm_model, tokenizer, candidate_config, query):
    """
    Query the LLM for an explanation of why a specific design was chosen.

    Args:
        llm_model: The LLM model instance.
        tokenizer: The tokenizer for the LLM model.
        candidate_config (dict): The candidate architecture configuration.
        query (str): The explanation prompt.

    Returns:
        str: Explanation generated by the LLM.
    """
    input_prompt = f"Candidate Configuration: {candidate_config}\n\n{query}"
    input_ids = tokenizer.encode(input_prompt, return_tensors="pt").to(llm_model.device)
    output_ids = llm_model.generate(input_ids, max_length=1024, num_return_sequences=1)
    explanation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return explanation

# ===============================
#      Candidate Management
# ===============================

def save_candidate(candidate_config, metrics, filename="best_candidate.json"):
    """
    Save the candidate configuration and metrics to a file.

    Args:
        candidate_config (dict): Network configuration of the candidate.
        metrics (dict): Performance metrics of the candidate.
        filename (str): Name of the file to save the candidate.
    """
    os.makedirs("saved_candidates/feed/", exist_ok=True)
    filepath = os.path.join("saved_candidates/feed/", filename)
    data = {
        "candidate_config": candidate_config,
        "metrics": metrics
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Candidate saved to {filepath}")


# ===============================
#      Main Function
# ===============================

def main():
    """
    Main function to perform LLM-based search.

    """
    # Example configurations for prompts
    json_exp_b5 = {
    "candidate_config": {
        "input_channels": 3,
        "blocks": [
            {
                "output_channels": 32,
                "num_layers": 1,
                "kernel_size": 3,
                "stride": 2,
                "expansion_factor": 2,
                "use_se": False,
                "se_ratio": 0.25,
                "conv_type": "mbconv",
                "skip_op": "residual",
                "activation": "relu6"
            },
            {
                "output_channels": 16,
                "num_layers": 3,
                "kernel_size": 3,
                "stride": 2,
                "expansion_factor": 2,
                "use_se": True,
                "se_ratio": 0.25,
                "conv_type": "mbconv",
                "skip_op": "residual",
                "activation": "relu6"
            },
            {
                "output_channels": 48,
                "num_layers": 2,
                "kernel_size": 5,
                "stride": 1,
                "expansion_factor": 2,
                "use_se": True,
                "se_ratio": 0,
                "conv_type": "mbconv",
                "skip_op": "residual",
                "activation": "leakyrelu"
            },
            {
                "output_channels": 64,
                "num_layers": 4,
                "kernel_size": 3,
                "stride": 1,
                "expansion_factor": 4,
                "use_se": False,
                "se_ratio": 0.25,
                "conv_type": "mbconv",
                "skip_op": "identity",
                "activation": "relu6"
            },
            {
                "output_channels": 64,
                "num_layers": 1,
                "kernel_size": 5,
                "stride": 2,
                "expansion_factor": 3,
                "use_se": False,
                "se_ratio": 0,
                "conv_type": "mbconv",
                "skip_op": "identity",
                "activation": "relu6"
            }
        ]
    }
    }
    # Define search space
    search_space = hierarchical_search_space()

    # CIFAR100 DataLoader with 160x160 images
    train_loader, val_loader, test_loader = load_cifar100(batch_size=128, image_size=160)

    # Perform feedback loop for network search
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize LLM
    llm_model, tokenizer = initialize_llm()

    best_candidate, pareto_set, history = feedback_loop_single_proposal(
        llm_model=llm_model,
        tokenizer=tokenizer,
        search_space=search_space,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        answer_exp=json_exp_b5,
        initial_in_channels=3, 
        num_blocks=5,  
        max_iterations=500,
        experiment_name="search_llama8b",
        mac_limit=350
    )
    print("\nBest Candidate Configuration:", best_candidate)

   
if __name__ == "__main__":
    main()

