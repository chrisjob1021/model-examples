# Model Examples Collection

This repository contains a collection of machine learning and deep learning model examples implemented in PyTorch. Each example is self-contained and designed for educational and reference purposes.

## Examples

- **Pointer Network** ([lstm/pointer-network.ipynb](lstm/pointer-network.ipynb))
  - Demonstrates how to use a Pointer Network to solve a toy sorting task.
  - Inspired by the paper: [Pointer Networks (Vinyals et al., 2015) - arXiv:1506.03134](https://arxiv.org/abs/1506.03134)
- **Bahdanau Attention NMT** ([attention/bahdanau-attention.ipynb](attention/bahdanau-attention.ipynb))
  - Toy sequence-to-sequence model with Bahdanau attention that reverses sequences.
  - Based on the paper: [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014) - arXiv:1409.0473](https://arxiv.org/abs/1409.0473)

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
   - For the pointer network example, open `lstm/pointer-network.ipynb`.
   - For the Bahdanau attention example, open `attention/bahdanau-attention.ipynb`.

3. **Run the notebook cells:**
   - Follow the instructions and run the cells to see the model in action.

## References
- [Pointer Networks (Vinyals et al., 2015) - arXiv:1506.03134](https://arxiv.org/abs/1506.03134)
- [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014) - arXiv:1409.0473](https://arxiv.org/abs/1409.0473)

## License

This project is licensed under the [MIT License](LICENSE). 