# Self-Driving Model with Convolutional Social Pooling
## Project Plan & Architecture

### Overview
Building a vehicle trajectory prediction model inspired by the Convolutional Social Pooling paper using the Waymo Open Dataset. The model will predict future vehicle trajectories by learning interdependencies in vehicle motion through an LSTM encoder-decoder architecture with convolutional social pooling.

### Dataset
- **Location**: `~/waymo/`
- **Structure**: 
  - Training: 1000 TFRecord files 
  - Testing: 150 TFRecord files
- **Format**: TensorFlow TFRecord format containing sensor data, labels, and metadata

### Architecture Components

#### 1. LSTM Encoder-Decoder Framework
- **Encoder**: Processes historical trajectory sequences
- **Decoder**: Generates future trajectory predictions
- **Hidden State Size**: 128-256 units (configurable)

#### 2. Convolutional Social Pooling Layer
- Captures spatial relationships between vehicles
- Constructs social tensor representing vehicle interactions
- Uses CNN to extract spatial features from social context
- Grid size: 13x3 cells (covering surrounding area)

#### 3. Input/Output Specifications
- **Input**: 
  - Past trajectory: 3 seconds (30 frames at 10Hz)
  - Vehicle states: position (x, y), velocity, heading
  - Social context: neighboring vehicles within 50m radius
- **Output**:
  - Future trajectory: 5 seconds (50 frames)
  - Multi-modal distribution over possible maneuvers
  - Uncertainty estimates

### Implementation Phases

#### Phase 1: Data Pipeline
- Parse Waymo TFRecord files
- Extract vehicle trajectories and interactions
- Create training/validation splits
- Implement data augmentation (rotation, translation)

#### Phase 2: Model Implementation
- Base LSTM encoder-decoder
- Social tensor construction
- Convolutional pooling layers
- Multi-modal prediction head

#### Phase 3: Training Infrastructure
- Loss functions: NLL + L2 trajectory loss
- Optimizer: Adam with learning rate scheduling
- Checkpointing and resume capability
- TensorBoard logging

#### Phase 4: Visualization
- Input visualization: Bird's eye view with vehicle positions
- Trajectory predictions with uncertainty
- Social interaction heatmaps
- Real-time inference demo

### Technical Stack
- **Framework**: PyTorch
- **Data Loading**: TensorFlow (for TFRecord parsing) + PyTorch DataLoader
- **Visualization**: Matplotlib, OpenCV
- **Logging**: TensorBoard, Weights & Biases (optional)

### Training Configuration
- **Batch Size**: 32-64
- **Learning Rate**: 1e-3 with cosine annealing
- **Epochs**: 50-100
- **Hardware**: GPU recommended (CUDA)

### Evaluation Metrics
- Average Displacement Error (ADE)
- Final Displacement Error (FDE)
- Negative Log-Likelihood (NLL)
- Collision rate
- Lane keeping accuracy

### Checkpoint Strategy
- Save every 5 epochs
- Keep best 3 models based on validation loss
- Include optimizer state for resume
- Metadata: epoch, loss history, hyperparameters

### Directory Structure
```
selfdriving/
├── PROJECT_PLAN.md (this file)
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── waymo_loader.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── csp_model.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   └── social_pooling.py
│   ├── training/
│   │   ├── train.py
│   │   ├── losses.py
│   │   └── metrics.py
│   └── visualization/
│       ├── visualizer.py
│       └── inference_demo.py
├── configs/
│   └── default_config.yaml
├── checkpoints/
├── logs/
└── outputs/
```

### Next Steps
1. Set up Python environment and dependencies
2. Implement Waymo data loader
3. Build core model architecture
4. Create training loop
5. Develop visualization tools
6. Run experiments and tune hyperparameters