# DA6401_A2

This repository contains the implementation of an assignment for the DA6401 course. The repository is structured into two main parts, `PartA` and `PartB`, each containing Jupyter notebooks and Python scripts for different tasks. Below is an overview of the repository structure and functionality.

---

## Repository Structure

### PartA
- **Files:**
  - `CS24M034_A2_A.ipynb`: A Jupyter Notebook containing the implementation of tasks for PartA. It includes analysis, visualizations, and explanations.
  - `cs24m034-A2_A.py`: A Python script implementing various functionalities such as data loading, model training, and visualization using PyTorch and WandB.

- **Key Highlights:**
  - Implements a convolutional neural network (CNN) for classification tasks.
  - Includes functions for data augmentation, visualization of predictions, and model training with mixed precision.
  - Supports hyperparameter tuning through WandB sweeps.

---

### PartB
- **Files:**
  - `cs24m034-A2_B.ipynb`: A Jupyter Notebook containing the implementation of tasks for PartB. Similar to PartA, it includes detailed explanations and visualizations.
  - `cs24m034-A2_B.py`: A Python script implementing deep learning tasks using transfer learning with PyTorch and WandB.

- **Key Highlights:**
  - Uses a pre-trained GoogLeNet model for transfer learning.
  - Provides options for feature extraction, freezing specific layers, and fine-tuning.
  - Implements data loaders optimized for GPU utilization and mixed precision training.

---

## Key Features

1. **Data Handling and Augmentation**:
   - Data is loaded and split into training, validation, and test sets.
   - Supports data augmentation through transformations like resizing, flipping, and rotation.

2. **Model Training**:
   - Supports custom CNNs (PartA) and transfer learning (PartB).
   - Includes mixed precision training for faster computation on GPUs.
   - Tracks training, validation, and test performance using accuracy and loss metrics.

3. **Hyperparameter Tuning**:
   - Utilizes WandB sweeps for optimizing parameters like batch size, dropout, and activation function.

4. **Visualization**:
   - Displays predictions along with true labels for test data.
   - Logs images and metrics to WandB for detailed analysis.

---

## Dependencies

The following Python libraries are required to run the code in this repository:
- `torch`: For deep learning model implementation.
- `torchvision`: For datasets and transforms.
- `wandb`: For logging and hyperparameter sweeps.
- `matplotlib`: For visualizations.
- `numpy`: For data manipulation.
- `pathlib`: For handling file paths.
- `tqdm`: For progress bars.

---

## Usage

1. **Data Preparation**:
   - Ensure that the dataset is organized into `train` and `val` directories.
   - Update the `dataset_path` variable in the scripts to point to your dataset location.

2. **Run Jupyter Notebooks**:
   - Use the `.ipynb` files for step-by-step explanations and results.

3. **Run Python Scripts**:
   - Execute `cs24m034-A2_A.py` or `cs24m034-A2_B.py` for end-to-end training and evaluation.

4. **WandB Integration**:
   - Log in to WandB using your API key to enable logging and hyperparameter sweeps.

---

## Results

Detailed results and metrics are logged to WandB during training. Visualizations, including predictions and training curves, are available in the notebooks.

---

## Contact

For any queries, please contact [Prateek](https://github.com/prateek122000).

```
