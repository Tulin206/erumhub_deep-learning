# Simple Deep Learning Implementation Guide

This guide explains the implementation of a basic deep learning model using PyTorch. The code demonstrates fundamental concepts including neural network architecture, training, validation, and visualization.

## Code Structure Overview

The implementation consists of several key components:

1. Neural Network Architecture (`SimpleNet` class)
2. Training and Validation Function (`train_model`)
3. Visualization Function (`plot_metrics`)
4. Main Execution Function (`main`)

## Implementation Steps

### Step 1: Required Libraries
- torch: Main PyTorch library
- torch.nn: Neural network modules
- torch.optim: Optimization algorithms
- torchvision: For accessing the MNIST dataset
- matplotlib: For visualization

### Step 2: Random Seed
Setting a random seed (42) ensures reproducible results across different runs.

## Detailed Component Explanation

### 1. SimpleNet Class
```python
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
```
- **Purpose**: Defines the neural network architecture
- **Components**:
  - `nn.Flatten()`: Converts 2D images (28x28) to 1D vectors (784)
  - First Linear Layer: 784 → 128 neurons
  - ReLU Activation: Adds non-linearity
  - Output Layer: 128 → 10 neurons (one for each digit)
- **Forward Method**: Defines how data flows through the network

### 2. Training Function
```python
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
```
- **Purpose**: Handles the training and validation process
- **Parameters**:
  - `model`: The neural network to train
  - `train_loader`: DataLoader for training data
  - `val_loader`: DataLoader for validation data
  - `criterion`: Loss function (CrossEntropyLoss)
  - `optimizer`: Optimization algorithm (Adam)
  - `num_epochs`: Number of training iterations
- **Process**:
  1. Training Phase:
     - Forward pass: Compute predictions
     - Calculate loss
     - Backward pass: Compute gradients
     - Update weights
  2. Validation Phase:
     - Evaluate model performance
     - No gradient computation needed
- **Returns**: Training and validation metrics (losses and accuracies)

### 3. Plotting Function
```python
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
```
- **Purpose**: Visualizes training progress
- **Plots**:
  - Training vs Validation Loss
  - Training vs Validation Accuracy
- **Helps**: Monitor model performance and detect issues like overfitting

### 4. Main Function
```python
def main():
```
- **Purpose**: Sets up and executes the training pipeline
- **Components**:
  1. Hyperparameter Definition:
     - `batch_size = 64`: Number of samples per training step
     - `learning_rate = 0.001`: Controls how much to adjust weights
     - `num_epochs = 10`: Number of complete dataset passes
  2. Data Preprocessing:
     - Loads MNIST dataset
     - Applies normalization
     - Splits into training (80%) and validation (20%) sets
  3. Model Setup:
     - Initializes SimpleNet
     - Defines loss function and optimizer
  4. Training Execution and Visualization

### Data Preprocessing Details
The implementation includes specific data preprocessing steps:
- Normalization using MNIST mean (0.1307) and standard deviation (0.3081)
- Dataset splitting: 80% training, 20% validation
- Data transformation using `torchvision.transforms`:
  - Convert to tensor
  - Apply normalization
- Batch processing with DataLoader
  - Batch size: 64
  - Shuffling enabled for training data

### Understanding `if __name__ == '__main__':`

```python
if __name__ == '__main__':
    main()
```
- **Purpose**: Entry point of the script
- **Explanation**:
  - In Python, each file has a special variable `__name__`
  - When running the file directly: `__name__` equals `'__main__'`
  - When importing the file: `__name__` equals the filename
  - This structure ensures the `main()` function only runs when the file is executed directly
  - Useful for: 
    - Making code both importable and executable
    - Preventing code from running when imported as a module

## Usage Instructions

1. **Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Running the Script**:
   ```bash
   python simple_deep_learning.py
   ```

3. **Expected Output**:
   - Training progress for each epoch
   - Two plots showing:
     - Training/Validation Loss
     - Training/Validation Accuracy

## Understanding the Results

1. **Loss Curves**:
   - Should generally decrease over time
   - If validation loss increases while training loss decreases: Overfitting

2. **Accuracy Curves**:
   - Should increase over time
   - Final validation accuracy indicates model performance
   - Gap between training and validation accuracy shows generalization

## Tips for Modification

1. **Adjusting Network Architecture**:
   - Modify `SimpleNet` class
   - Add/remove layers
   - Change layer sizes

2. **Tuning Hyperparameters**:
   - Adjust `batch_size` for memory/speed trade-off
   - Modify `learning_rate` for stability/speed
   - Change `num_epochs` for training duration

3. **Data Preprocessing**:
   - Modify `transform` for different data augmentation
   - Adjust train/validation split ratio

## Common Issues and Solutions

1. **High Training Loss**:
   - Increase model capacity (more layers/neurons)
   - Adjust learning rate
   - Check for data preprocessing issues

2. **High Validation Loss**:
   - Add regularization
   - Reduce model complexity
   - Increase training data

3. **Slow Training**:
   - Increase batch size
   - Check if GPU is being used
   - Simplify model architecture
