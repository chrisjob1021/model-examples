  1. No normalization layers - The network lacks batch normalization, which is critical for deep networks
  2. No residual connections - Deep layers (conv2: 4 layers, conv3: 6 layers) suffer from gradient vanishing
  3. Small 2×2 kernels - Conv2 and conv3 use tiny 2×2 kernels, limiting receptive field growth
  5. Limited depth - Only ~10 conv layers total, modern architectures use 50-100+ layers