# Model Examples Collection

This repository contains a collection of machine learning and deep learning model examples implemented in PyTorch. Each example is self-contained and designed for educational and reference purposes.

## Examples

- **Pointer Network** ([lstm/pointer-network.ipynb](lstm/pointer-network.ipynb))
  - Demonstrates how to use a Pointer Network to solve a toy sorting task.
  - Inspired by the paper: [Pointer Networks (Vinyals et al., 2015) - arXiv:1506.03134](https://arxiv.org/abs/1506.03134)
- **PReLU CNN** ([prelu/prelu-example.ipynb](prelu/prelu-example.ipynb))
  - Shows how to use the Parametric ReLU activation in a simple convolutional network.
  - Includes an optional section demonstrating training with the HuggingFace Trainer.

More model examples will be added to this repository over time.

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/model-examples.git
   cd model-examples
   ```

2. **(Optional) Create and activate a virtual environment:**
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install torch jupyter
   ```
   You may need to install additional dependencies as new examples are added.

### Running the Example Notebooks

1. **Start Jupyter Notebook:**
   ```sh
   jupyter notebook
   ```

2. **Open the desired notebook:**
   - For the pointer network example, navigate to `lstm/pointer-network.ipynb` in the Jupyter interface and open it.

3. **Run the notebook cells:**
   - Follow the instructions and run the cells to see the model in action.

## References
- [Pointer Networks (Vinyals et al., 2015) - arXiv:1506.03134](https://arxiv.org/abs/1506.03134)

## License

This project is licensed under the [MIT License](LICENSE). 