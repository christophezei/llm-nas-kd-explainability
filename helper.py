import json
import os
import torch
from ptflops import get_model_complexity_info

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_best_candidate(filepath="models/Llama8b/LMaNet_elite.json"):
    """
    Load the best candidate configuration and metrics from a file.

    Args:
        filepath (str): Path to the saved candidate file.

    Returns:
        tuple: Candidate configuration and metrics.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No saved candidate found at {filepath}")

    with open(filepath, "r") as f:
        data = json.load(f)

    print(f"Loaded best candidate from {filepath}")
    return data["candidate_config"], data["metrics"]


def compute_model_stats(model, input_shape=(3, 160, 160)):
    """
    Computes the number of parameters, model size, and MACs using ptflops.

    Args:
        model (nn.Module): The model to analyze.
        input_shape (tuple): The shape of the input tensor (C, H, W).

    Returns:
        dict: Model statistics including number of parameters (millions), model size (MB), and MACs (millions).
    """
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = num_params * 4 / (1024 ** 2)  # Assuming 32-bit floats, in MB

    try:
        macs, _ = get_model_complexity_info(model, input_shape, as_strings=False, print_per_layer_stat=False)
        macs_millions = macs / 1e6  # Convert to millions
    except Exception as e:
        print(f"Error computing MACs: {e}")
        macs_millions = None

    return {
        'num_params_millions': num_params / 1e6,
        'model_size_MB': model_size,
        'macs_millions': macs_millions
    }

def compute_peak_sram(model, input_shape, dtype=torch.int8):
    """
    Compute a refined peak SRAM usage estimation for a PyTorch model during inference.

    Args:
        model (torch.nn.Module): The PyTorch model.
        input_shape (tuple): Shape of the input tensor (batch_size, channels, height, width).
        dtype (torch.dtype): Data type for activations (default: torch.int8).

    Returns:
        float: Estimated peak SRAM usage in MB.
    """
    model.to(device)
    dtype_size = torch.tensor([], dtype=dtype).element_size()  # Size of each element in bytes
    peak_memory = 0  # Peak memory usage
    current_memory = 0  # Current memory usage
    
    # Memory for input tensor
    input_tensor_size = torch.prod(torch.tensor(input_shape)) * dtype_size
    current_memory += input_tensor_size.item()
    peak_memory = max(peak_memory, current_memory)

    def hook(module, input, output):
        nonlocal current_memory, peak_memory
        # Account for the memory of output activations
        output_size = output.numel() * dtype_size
        current_memory += output_size
        peak_memory = max(peak_memory, current_memory)
        
        # Simulate releasing memory for intermediate outputs
        current_memory -= output_size

    # Register hooks on all layers
    hooks = []
    for layer in model.modules():
        hooks.append(layer.register_forward_hook(hook))
    
    # Perform a forward pass with a dummy input
    dummy_input = torch.randn(input_shape, dtype=torch.float32).to(device)  # Assume float input
    model(dummy_input)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Convert peak memory to MB
    peak_memory_mb = peak_memory / (1024 ** 2)
    return peak_memory_mb
