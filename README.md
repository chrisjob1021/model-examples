# Model Examples Collection

This repository contains a collection of machine learning and deep learning model examples implemented in PyTorch. Each example is self-contained and designed for educational and reference purposes.

## Examples

- **Pointer Network** ([lstm/pointer-network.ipynb](lstm/pointer-network.ipynb))
  - Demonstrates how to use a Pointer Network to solve a toy sorting task.
  - Inspired by the paper: [Pointer Networks (Vinyals et al., 2015) - arXiv:1506.03134](https://arxiv.org/abs/1506.03134)
- **RNNsearch** ([rnnsearch/toy_rnnsearch.py](rnnsearch/toy_rnnsearch.py))
  - Minimal implementation of the attention-based model from [Bahdanau et al., 2014 - arXiv:1409.0473](https://arxiv.org/pdf/1409.0473.pdf)
- **Convolutional Neural Network** ([cnn/cnn.ipynb](cnn/cnn.ipynb))
  - Basic CNN example for image classification.

More model examples will be added to this repository over time.

## Gradients

- [gradients/lstm.ipynb](gradients/lstm.ipynb) - Explore LSTM gradients.

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/model-examples.git
   cd model-examples
   ```

2. **Set up a virtual environment and install dependencies:**
   ```sh
   ./scripts/setup_venv.sh
   ```
   The script creates a `.venv` folder and installs the packages listed in
   `requirements.txt`.

### Running the Example Notebooks

1. **Start Jupyter Notebook:**
   ```sh
   jupyter notebook
   ```

2. **Open the desired notebook:**
   - For the pointer network example, navigate to `lstm/pointer-network.ipynb` in the Jupyter interface and open it.
   - For the CNN example, open `cnn/cnn.ipynb`.

3. **Run the notebook cells:**
   - Follow the instructions and run the cells to see the model in action.

## References
- [Pointer Networks (Vinyals et al., 2015) - arXiv:1506.03134](https://arxiv.org/abs/1506.03134)
- [Generating Sequences With Recurrent Neural Networks (Graves, 2014) - arXiv:1409.0473](https://arxiv.org/pdf/1409.0473)

## License

This project is licensed under the [MIT License](LICENSE). 
