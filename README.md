# Model Examples Collection

This repository contains a collection of machine learning and deep learning model examples implemented in PyTorch. Each example is self-contained and designed for educational and reference purposes.

## Examples

- **Pointer Network** ([lstm/pointer-network.ipynb](lstm/pointer-network.ipynb))
  - Demonstrates how to use a Pointer Network to solve a toy sorting task.
  - Inspired by the paper: [Pointer Networks (Vinyals et al., 2015) - arXiv:1506.03134](https://arxiv.org/abs/1506.03134)
- **RNNsearch** ([rnnsearch/rnnsearch.py](rnnsearch/rnnsearch.py))
  - Minimal implementation of the attention-based model from [Bahdanau et al., 2014 - arXiv:1409.0473](https://arxiv.org/pdf/1409.0473.pdf)
- **Convolutional Neural Network with PReLU** ([cnn/](cnn/))
  - Implementation of PReLU (Parametric Rectified Linear Unit) from [He et al., 2015 - arXiv:1502.01852](https://arxiv.org/abs/1502.01852)
  - Complete training pipeline for ImageNet-1k classification
  - Educational features including manual convolution/pooling implementations
  - Activation visualization tools to understand what CNNs learn
  - See [cnn/README.md](cnn/README.md) for detailed documentation

## Gradients

- [gradients/lstm.ipynb](gradients/lstm.ipynb) - Explore LSTM gradients.

## Running The Examples

### Setup

1. **Set up a virtual environment and install dependencies:**
   ```sh
   ./scripts/setup_venv.sh
   ```
   The script creates a `.venv` folder and installs the packages listed in
   `requirements.txt`.

### Running Notebooks

1. **Start Jupyter Notebook:**
   ```sh
   jupyter notebook
   ```

2. **Open the desired notebook:**
   - For the pointer network example, navigate to `lstm/pointer-network.ipynb` in the Jupyter interface and open it.
   - For CNN examples, see notebooks in the `cnn/` directory.

3. **Run the notebook cells:**
   - Follow the instructions and run the cells to see the model in action.

### Training Models

Some examples include full training pipelines:

1. **CNN on ImageNet:**
   ```sh
   cd cnn
   python train_cnn_imagenet.py
   ```
   See [cnn/README.md](cnn/README.md) for detailed instructions.

### Monitoring Training

Use TensorBoard to visualize training progress:
```sh
./scripts/start_tensorboard.sh
```

## License

This project is licensed under the [MIT License](LICENSE). 
