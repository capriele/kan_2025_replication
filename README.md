# KAN 2025 Replication Package

Welcome to the KAN 2025 replication package repository! This repository provides all the necessary resources—data, models, and code—to reproduce the experiments and results from the KAN 2025 project.

## Getting Started

Follow the steps below to set up your environment and begin working with the project.

### Prerequisites

Ensure you have the following installed before proceeding:

- **Python 3.10.8** (recommended)
- **[`pyenv`](https://github.com/pyenv/pyenv)** for Python version management
- **[`pip`](https://pip.pypa.io/en/stable/)** for package installation

### Installation

1. **Install `pyenv` (if not already installed):**  
   Follow the official instructions for your OS:  
   - **Linux/macOS:**
     ```sh
     curl https://pyenv.run | bash
     ```
     Follow the instructions to update your shell configuration (`~/.bashrc`, `~/.zshrc`, etc.).
   - **Windows:** Use [`pyenv-win`](https://github.com/pyenv-win/pyenv-win).

2. **Install Python 3.10.8 using `pyenv`:**
   ```sh
   pyenv install 3.10.8
    ```

3. **Create and activate a virtual environment (pykan):**
    ```sh
    pyenv virtualenv 3.10.8 pykan
    pyenv activate pykan
    ```

4. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/kan_2025_replication_package.git
    cd kan_2025_replication_package
    ```

5. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training

To train the anomaly detection model, run the `kan_anomaly_detection_training.ipynb` notebook.

### Testing

To test the anomaly detection model, run the `kan_anomaly_detection_testing.ipynb` notebook.

### Boundary Validation

To validate the model's performance on safe boundary detection, use the `test_boundary_validation.ipynb` notebook.

## Directory Details

- data: Contains the training and testing data files.
- kan: Contains the core code for the KAN project, including various modules and utilities.
- model: Contains the saved model states and configurations.
- `kan_anomaly_detection_training.ipynb`: Jupyter notebook for training the anomaly detection model.
- `kan_anomaly_detection_testing.ipynb`: Jupyter notebook for testing the anomaly detection model.
- `test_boundary_validation.ipynb`: Jupyter notebook for validating safe boundary detection.

## Contributing

We welcome contributions! If you have improvements or bug fixes, please open an issue or submit a pull request.

## License

This project is licensed under the Apache 2.0 License.

## Acknowledgments

- Special thanks to the contributors of the KAN project.
- This project is based on the work presented in the KAN 2025 paper.