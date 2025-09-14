# Hidden Layers and Neurons Analysis using Keras Tuner

This implementation demonstrates how to use Keras Tuner to automatically find the optimal number of hidden layers and neurons for neural networks. The code includes two different implementations:
1. Air Quality Prediction (Real_Combine.csv)
2. MNIST Digit Classification

## Implementations Overview

### 1. Air Quality Prediction

The notebook uses Keras Tuner's RandomSearch functionality to experiment with different neural network architectures by varying:
1. Number of hidden layers
2. Number of neurons in each layer
3. Learning rate

### 2. MNIST Digit Classification

The notebook uses Keras Tuner's RandomSearch functionality to find optimal architecture for MNIST digit classification by varying:
1. Number of hidden layers (1-3)
2. Number of neurons per layer (32-512)
3. Dropout rates (0-0.5)
4. Learning rate (1e-4 to 1e-2)

## Datasets

1. **Real_Combine.csv**

The dataset contains various air quality metrics and environmental factors. The model aims to predict air quality based on these features.

2. **MNIST Dataset**
   - 70,000 grayscale images of handwritten digits (0-9)
   - Each image is 28x28 pixels
   - Split into 60,000 training and 10,000 test images
   - Automatically downloaded through keras.datasets

## Code Structure

### 1. Setup and Imports
```python
# For Air Quality Implementation
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

# Additional imports for MNIST Implementation
import tensorflow as tf
import keras_tuner
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

### 2. Data Preparation

#### Air Quality Data
- Load the dataset using pandas
- Split features into:
  - X (independent features): All columns except the last
  - y (dependent feature): Last column (air quality metric)
- Split data into training (70%) and testing (30%) sets

#### MNIST Data
- Load MNIST dataset using keras.datasets
- Preprocess images:
  - Reshape 28x28 images to 784-dimensional vectors
  - Normalize pixel values to [0-1] range
- Convert labels to one-hot encoding (10 classes)
- Split into training and validation sets (80-20%)

### 3. Model Architecture Search Space

#### Air Quality Model

The `build_model` function defines the search space for the neural network architecture:

```python
def build_model(hp):
    model = keras.Sequential()
    # Number of layers: 2 to 20
    for i in range(hp.Int('num_layers', 2, 20)):
        # Number of neurons: 32 to 512 (step of 32)
        model.add(layers.Dense(
            units=hp.Int('units_' + str(i),
                        min_value=32,
                        max_value=512,
                        step=32),
            activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    
    # Learning rate options: 0.01, 0.001, or 0.0001
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error'])
    return model
```

#### MNIST Model
```python
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(784,)))
    
    # Tune number of layers (1-3)
    n_layers = hp.Int('num_layers', min_value=1, max_value=3)
    
    for i in range(n_layers):
        # Tune units per layer (32-512)
        units = hp.Int(f'units_layer_{i}', min_value=32, max_value=512, step=32)
        model.add(keras.layers.Dense(units=units, activation='relu'))
        
        # Tune dropout rate (0-0.5)
        dropout_rate = hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)
        model.add(keras.layers.Dropout(dropout_rate))
    
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    # Tune learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

### 4. Training Configuration

#### Air Quality Model

The RandomSearch tuner is configured with:
- Maximum 5 trials
- 3 executions per trial
- Optimization objective: validation mean absolute error
- Results stored in 'project/Air Quality Index' directory

#### MNIST Model
- Maximum 5 trials
- Early stopping with 5 epochs patience
- Learning rate reduction on plateau
- Batch size: 128
- Training epochs: 10 (search), 20 (final model)
- Validation split: 20%
- Results stored in 'keras_tuner/mnist_tuning' directory

## Hyperparameters Explored

1. **Number of Hidden Layers**
   - Range: 2 to 20 layers (air quality) or 1 to 3 layers (MNIST)
   - Each layer uses ReLU activation

2. **Neurons per Layer**
   - Range: 32 to 512 neurons
   - Step size: 32
   - Each layer can have a different number of neurons

3. **Learning Rate**
   - Options: 0.01, 0.001, 0.0001 (air quality) or 1e-4 to 1e-2 (MNIST)
   - Uses Adam optimizer

4. **Dropout Rate (MNIST only)**
   - Range: 0 to 0.5
   - Helps prevent overfitting

## Usage

1. **Setup Requirements**
   ```bash
   cd "Hidden Layer Analysis"
   pip install -r requirements.txt  # For air quality prediction
   pip install -r simple_requirements.txt  # For MNIST implementation
   ```

2. **Run the Notebooks**
   - `Hidden Layers And Hidden Neurons.ipynb` for air quality prediction
   - `simple_keras_tuner_mnist.ipynb` for MNIST classification

## Results Analysis

The tuner provides:
1. Search space summary before training
2. Results summary after training, including:
   - Best model architecture
   - Optimal number of layers
   - Optimal number of neurons per layer
   - Best learning rate
   - Validation mean absolute error

### MNIST-Specific Results
- Test accuracy on unseen data
- Best model architecture details:
  - Optimal number of layers
  - Units per layer
  - Dropout rates
  - Learning rate
- Saved model file (best_mnist_model.keras)

## Project Structure

```
Hidden Layer Analysis/
├── Hidden Layers And Hidden Neurons.ipynb  # Air quality implementation
├── simple_keras_tuner_mnist.ipynb         # MNIST implementation
├── hidden_layers_&_hidden_neurons.py      # Python script for air quality
├── simple_keras_tuner_mnist.py            # Python script for MNIST
├── Real_Combine.csv                       # Air quality dataset
├── requirements.txt                       # Main requirements
├── simple_requirements.txt                # MNIST-specific requirements
└── README.md                             # This documentation

keras_tuner/                              # Results directory
├── mnist_tuning/                         # MNIST tuning results
└── simple_tuning/                        # Air quality tuning results
```
