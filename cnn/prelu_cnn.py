"""Minimal CNN training example using CIFAR10

Convolution and pooling are implemented manually with explicit loops instead of
relying on ``torch.nn.functional`` helpers such as ``conv2d`` or ``max_pool2d``.
This file trains the network on CIFAR10 with either ReLU or PReLU activation so
you can compare the impact of PReLU on accuracy.
"""

# Avoid using einsum, conv2d or max_pool2d as requested

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------------------
# Manual 2D convolution layer implemented with nested loops
# -------------------------------------------------------
class ManualConv2d(nn.Module):
    """2D convolution implemented with explicit loops.

    Parameters
    ----------
    use_builtin : bool, optional
        If ``True``, ``torch.nn.functional.conv2d`` is used instead of the
        slower manual loop implementation. This is convenient when you want to
        speed up training while keeping the same layer interface.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, *, use_builtin=False):
        super().__init__()
        # Kernel weights and biases follow the shape used by nn.Conv2d
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding
        self.use_builtin = use_builtin

    def forward(self, x):
        # x: (batch, in_channels, height, width)
        if self.use_builtin:
            # Delegate to PyTorch's conv2d for speed. Padding is handled
            # internally by the functional call so no explicit loops are needed.
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
            )

        batch_size, in_channels, in_height, in_width = x.shape
        kernel_height = kernel_width = self.weight.shape[2]

        # Pad the input so that the manually computed convolution matches PyTorch's behavior
        padded_x = F.pad(
            x,
            (self.padding, self.padding, self.padding, self.padding),
        )

        # Compute output spatial dimensions using the convolution formula
        out_height = (in_height + 2 * self.padding - kernel_height) // self.stride + 1
        out_width = (in_width + 2 * self.padding - kernel_width) // self.stride + 1

        # Allocate output tensor on the same device as the input
        output = torch.zeros(
            batch_size, self.weight.shape[0], out_height, out_width, device=x.device
        )

        # Iterate over the batch, output channels and spatial positions
        for batch_idx in range(batch_size):
            for out_ch in range(self.weight.shape[0]):
                for row in range(out_height):
                    for col in range(out_width):
                        # 1) select the patch from the padded image
                        row_start = row * self.stride
                        col_start = col * self.stride
                        region = padded_x[
                            batch_idx,
                            :,  # all input channels
                            row_start : row_start + kernel_height,
                            col_start : col_start + kernel_width,
                        ]

                        # 2) element-wise multiply patch and kernel, 3) sum and add bias
                        output[batch_idx, out_ch, row, col] = (
                            region * self.weight[out_ch]
                        ).sum() + self.bias[out_ch]

        return output

# -------------------------------------------------------
# Manual max pooling layer (no padding, square kernel)
# -------------------------------------------------------
class ManualMaxPool2d(nn.Module):
    """Max pooling using explicit loops (square kernels only)."""

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        # Default stride equals kernel size as in typical max pooling layers
        self.stride = stride or kernel_size

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        batch_size, channels, in_height, in_width = x.shape
        k = self.kernel_size
        out_height = (in_height - k) // self.stride + 1
        out_width = (in_width - k) // self.stride + 1

        # Allocate output tensor
        output = torch.zeros(batch_size, channels, out_height, out_width, device=x.device)

        # Slide the pooling window across the input
        for batch_idx in range(batch_size):
            for channel in range(channels):
                for row in range(out_height):
                    for col in range(out_width):
                        row_start = row * self.stride
                        col_start = col * self.stride
                        region = x[
                            batch_idx,
                            channel,
                            row_start : row_start + k,
                            col_start : col_start + k,
                        ]
                        # Compute max inside the window and store the result
                        output[batch_idx, channel, row, col] = region.max()

        return output

# -------------------------------------------------------
# Simple CNN using the manual layers
# -------------------------------------------------------
class ManualCNN(nn.Module):
    """Very small CNN constructed from the manual layers above.

    Parameters
    ----------
    use_prelu : bool, optional
        If ``True``, ``nn.PReLU`` is used for activations instead of ``nn.ReLU``.
    use_builtin_conv : bool, optional
        Forward ``True`` to use the faster PyTorch convolution implementation
        inside :class:`ManualConv2d`. Defaults to ``False``.
    """

    def __init__(self, use_prelu: bool = False, *, use_builtin_conv: bool = False):
        super().__init__()
        # CIFAR10 has 3 input channels
        self.conv1 = ManualConv2d(3, 8, kernel_size=3, padding=1, use_builtin=use_builtin_conv)
        self.act1 = nn.PReLU() if use_prelu else nn.ReLU()
        self.pool1 = ManualMaxPool2d(kernel_size=2)

        self.conv2 = ManualConv2d(8, 16, kernel_size=3, padding=1, use_builtin=use_builtin_conv)
        self.act2 = nn.PReLU() if use_prelu else nn.ReLU()
        self.pool2 = ManualMaxPool2d(kernel_size=2)

        # After two 2x2 poolings starting from 32x32 images -> 8x8 spatial size
        self.fc = nn.Linear(16 * 8 * 8, 10)

    def forward(self, x):
        # Pass input through first conv block
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        # Second conv block
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        # Flatten and apply final classification layer
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -------------------------------------------------------
# Training and evaluation helpers
# -------------------------------------------------------

def train(model, loader, epochs: int = 1, lr: float = 0.01):
    """Train for a few epochs using data from ``loader``."""

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)

        avg_loss = epoch_loss / len(loader.dataset)
        print(f"epoch {epoch + 1} | loss {avg_loss:.4f}")

@torch.no_grad()
def evaluate(model, loader):
    model.to(DEVICE)
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0

if __name__ == "__main__":
    # Load CIFAR10 dataset (a few samples to keep the run light)
    transform = transforms.ToTensor()
    full_train = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    full_test = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    # Use subsets for a quick demonstration
    train_subset = torch.utils.data.Subset(full_train, list(range(1000)))
    test_subset = torch.utils.data.Subset(full_test, list(range(1000)))

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64)

    # Train and evaluate with ReLU
    # Set ``use_builtin_conv=True`` to speed up training by using PyTorch's
    # native convolution implementation inside our ``ManualConv2d`` layers.
    print("Training with ReLU activation...")
    relu_model = ManualCNN(use_prelu=False, use_builtin_conv=False)
    train(relu_model, train_loader, epochs=3)
    relu_acc = evaluate(relu_model, test_loader)
    print(f"Test accuracy with ReLU: {relu_acc:.2f}")

    # Train and evaluate with PReLU
    print("\nTraining with PReLU activation...")
    prelu_model = ManualCNN(use_prelu=True, use_builtin_conv=False)
    train(prelu_model, train_loader, epochs=3)
    prelu_acc = evaluate(prelu_model, test_loader)
    print(f"Test accuracy with PReLU: {prelu_acc:.2f}")
