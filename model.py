import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import load_best_candidate, compute_model_stats, compute_peak_sram

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction_ratio=0.25):
        super(SqueezeExcitation, self).__init__()
        reduced_channels = max(1, int(channels * reduction_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class InvertedResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, expansion_factor, use_se, se_ratio,
        conv_type='mbconv', skip_op='identity', activation='relu', dropout_rate=0.2
    ):
        super(InvertedResidualBlock, self).__init__()
        self.skip_op = skip_op
        self.use_res_connect = (skip_op == 'residual') and (stride == 1)
        self.use_se = use_se and se_ratio > 0

        # Activation Function
        activation_layer = self.get_activation(activation)

        if conv_type == 'depthwise':
            norm_layer = nn.BatchNorm2d(in_channels)  # Match input channels for depthwise
        elif conv_type == 'mbconv':
            norm_layer = nn.BatchNorm2d(in_channels * expansion_factor)  # Match expanded channels
        else:
            norm_layer = nn.BatchNorm2d(out_channels)  # Match output channels

        # Dropout Layer
        if dropout_rate > 0:
            dropout = nn.Dropout(p=dropout_rate)
        else:
            dropout = nn.Identity()

        # Main convolutional block
        if conv_type == 'mbconv':
            layers = [
                nn.Conv2d(in_channels, in_channels * expansion_factor, 1, 1, 0, bias=False),
                norm_layer,
                activation_layer,

                nn.Conv2d(
                    in_channels * expansion_factor,
                    in_channels * expansion_factor,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    groups=in_channels * expansion_factor,
                    bias=False
                ),
                nn.BatchNorm2d(in_channels * expansion_factor),
                activation_layer
            ]

            if self.use_se:
                layers.append(SqueezeExcitation(in_channels * expansion_factor, se_ratio))

            layers.extend([
                nn.Conv2d(in_channels * expansion_factor, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            ])

            self.conv = nn.Sequential(*layers)

        elif conv_type == 'depthwise':
            layers = [
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    groups=in_channels,
                    bias=False
                ),
                norm_layer,
                activation_layer
            ]

            if self.use_se:
                layers.append(SqueezeExcitation(in_channels, se_ratio))

            layers.extend([
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            ])

            self.conv = nn.Sequential(*layers)

        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")

        # Projection for residual connections if channels differ
        if self.use_res_connect and in_channels != out_channels:
            self.res_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.res_proj = None

    def forward(self, x):
        identity = x
        out = self.conv(x)

        if self.use_res_connect:
            if self.res_proj:
                identity = self.res_proj(identity)
            out += identity
        return out

    def get_activation(self, activation):
        """
        Returns the activation layer based on the activation string.
        """
        activation = activation.lower()
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'relu6':
            return nn.ReLU6(inplace=True)
        elif activation == 'leakyrelu':
            return nn.LeakyReLU(inplace=True)
        elif activation == 'swish':
            return nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation type: {activation}")


class NetworkBlock(nn.Module):
    def __init__(self, block_config, in_channels, out_channels):
        super(NetworkBlock, self).__init__()
        self.layers = nn.ModuleList()
        first_layer = InvertedResidualBlock(
            in_channels, out_channels,
            block_config['kernel_size'], block_config['stride'],
            block_config['expansion_factor'], block_config['use_se'],
            block_config['se_ratio'], block_config['conv_type'],
            block_config['skip_op'], block_config.get('activation')
        )
        self.layers.append(first_layer)

        for _ in range(1, block_config['num_layers']):
            self.layers.append(InvertedResidualBlock(
                out_channels, out_channels,
                block_config['kernel_size'], 1,
                block_config['expansion_factor'], block_config['use_se'],
                block_config['se_ratio'], block_config['conv_type'],
                block_config['skip_op'], block_config.get('activation')
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class LLMGenModel(nn.Module):
    def __init__(self, config, num_classes=100, dropout_rate=0.2, pool_type='average'):
        """
        Initialize the LLMGenModel using the provided blocks with an optional final layer.

        Args:
            config (dict): Configuration for the model, including blocks.
            num_classes (int): Number of output classes.
            dropout_rate (float): Dropout rate for regularization.
            pool_type (str): Type of pooling ('average', 'max', 'global').
            use_final_layer (bool): Whether to include the final layer.
        """
        super(LLMGenModel, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout_rate = dropout_rate

        # Input channels from the configuration
        in_channels = config['input_channels']

        # Build all blocks from the configuration
        for block_config in config['blocks']:
            block = NetworkBlock(block_config, in_channels, block_config['output_channels'])
            self.layers.append(block)
            in_channels = block_config['output_channels']

        # Add pooling layer based on pool_type
        if pool_type.lower() == 'average':
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pool_type.lower() == 'max':
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        elif pool_type.lower() == 'global':
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global pooling is typically average pooling
        else:
            raise ValueError(f"Unsupported pooling type: {pool_type}")

        # Add classifier
        self.classifier = nn.Linear(in_channels, num_classes)

        # Add dropout if specified
        self.dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

        # Initialize weights
        self.initialize_weights()

    def forward(self, x):
        # Pass through each block
        for layer in self.layers:
            x = layer(x)

        # Pooling
        x = self.global_pool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Apply dropout and pass through the classifier
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0 if hasattr(m, "residual") else 1)  # Zero γ for residual connections
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def main():
    # Define the path to the best candidate configuration and weights
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

    # Load the weights into the model
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cuda")
        model.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded model weights from {weights_path}")
    else:
        print(f"Model weights not found at {weights_path}")

    # Compute and print model stats
    print(model)
    model.eval()  # Set the model to evaluation mode
    stats = compute_model_stats(model, input_shape=(3, 160, 160))  # Assuming input shape of (3, 160, 160)
    print(f"Model Stats: Params: {stats['num_params_millions']:.2f}M, Size: {stats['model_size_MB']:.2f}MB, MACs: {stats['macs_millions']:.2f}M")

if __name__ == "__main__":
    main()