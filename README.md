# Deep Learning Project Setup and Usage Guide

This repository contains PyTorch and TensorFlow implementations of various deep learning models and exercises. The project provides two ways to work with the code:
- Python (.py) files for local development
- Jupyter Notebooks (.ipynb) for Google Colab usage

## Important Note About GPU Support

By default, the requirements files install CPU versions of PyTorch and TensorFlow. This ensures compatibility across all systems. The code will run on both CPU and GPU environments. If you want to use GPU acceleration:

- For PyTorch: Visit [PyTorch Installation](https://pytorch.org/get-started/locally/) to install the GPU version
- For TensorFlow: Visit [TensorFlow Installation](https://www.tensorflow.org/install/pip) to install the GPU version

## Environment Setup

This project uses [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#python-versions) for fast Python environment management and dependency installation.

Follow the installation instructions for `uv` at the link above.

## Local Development with Python Files

For local development, you should use the .py files provided in each directory. The .ipynb files are specifically prepared for Google Colab usage.

### Setting up PyTorch Environment

1. Navigate to the PyTorch directory:
   ```bash
   cd pytorch
   ```

2. Create and activate a virtual environment:
   ```bash
   uv venv .venv
   source .venv/bin/activate  # On Unix/macOS
   # Or on Windows:
   # .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

Available Python Files for Local Development:
- `simple_deep_learning.py` - A comprehensive implementation of deep learning fundamentals including:
  - Neural network architecture
  - Forward and backward propagation
  - Training and validation process
  - Loss computation and optimization
  - Performance visualization
  - Uses MNIST dataset (automatically downloaded to data/MNIST)

### Setting up TensorFlow Environment

1. Navigate to the TensorFlow directory:
   ```bash
   cd tensorflow
   ```

2. Create and activate a virtual environment:
   ```bash
   uv venv .venv
   source .venv/bin/activate  # On Unix/macOS
   # Or on Windows:
   # .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

TensorFlow Project Structure:

1. ANN Implementation/
   - Scripts: `ann.py` and `ann.ipynb`
   - Dataset: `Churn_Modelling.csv` (included in folder)
   - Purpose: Predicts if a bank customer will leave the bank (churn) based on various features
   - Features: Credit score, geography, gender, account balance, etc.
   - Documentation: Includes detailed README.md explaining the implementation

2. Hidden Layer Analysis/
   - Scripts: 
     - `hidden_layers_&_hidden_neurons.py` and `Hidden Layers And Hidden Neurons.ipynb` - Air quality prediction
     - `simple_keras_tuner_mnist.py` and `simple_keras_tuner_mnist.ipynb` - MNIST digit classification
   - Datasets: 
     - `Real_Combine.csv` (included in folder) - For air quality prediction
     - MNIST dataset (automatically downloaded) - For digit classification
   - Purpose: Experiments with different neural network architectures for both air quality prediction and image classification
   - Focus: 
     - Analysis of optimal number of hidden layers and neurons using Keras Tuner
     - Comparative study between different types of data (numerical vs image)
   - Documentation: Includes detailed README.md explaining the analysis process

3. Hyperparameter Tuning/
   - Scripts: `Hyperparameter.py` and `Hyperparameter.ipynb`
   - Dataset: `Real_Combine.csv` (included in folder)
   - Purpose: Predicts air quality metrics using optimized neural networks
   - Features: Hyperparameter tuning using Keras Tuner
   - Additional: Contains `project/Air Quality Index/` with tuning results and configurations
   - Documentation: Includes detailed README.md explaining the tuning process

Data Files:
- `Churn_Modelling.csv` - Banking customer data for churn prediction
  - Used by: ANN and Hidden Layers experiments
  - Features: Customer demographics, banking behavior
- `Real_Combine.csv` - Air quality measurement data
  - Used by: Hyperparameter tuning experiments
  - Features: Various air quality metrics and environmental factors

### Running Python Files Locally

After setting up your environment:

1. Activate the appropriate virtual environment (PyTorch or TensorFlow)
2. Run the desired Python file:
   ```bash
   python file_name.py
   ```

## Google Colab Usage with Jupyter Notebooks

If you prefer using Google Colab, you can use the provided .ipynb files which are specifically formatted for the Colab environment.

### Initial Setup in Google Colab

1. **Mount Google Drive:**
   - After uploading your notebook, you need to mount your Google Drive
   - Run the following code in a cell:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Click the authorization link and follow the prompts
   - After authorization, your Google Drive will be mounted and accessible

2. **Set Up Python Path:**
   - First, copy the notebook to your Google Drive's "Colab Notebooks" folder
   - Then run the following code to set up the correct paths:
     ```python
     import os, sys
     BASE = '/content/drive/MyDrive/Colab Notebooks'  # this can have spaces
     os.chdir(BASE)             # go into the folder
     sys.path.insert(0, BASE)   # add it to Python path
     ```

3. **Verify Setup:**
   - In the left sidebar, you should see "MyDrive"
   - Under MyDrive, look for the "Colab Notebooks" folder
   - Make sure your notebook is visible in this folder

### Available Notebooks

#### PyTorch Notebooks:
- `simple_deep_learning.ipynb` - Interactive notebook version of the deep learning implementation
  - Uses MNIST dataset (automatically downloaded)
  - Includes step-by-step explanations of each concept

#### TensorFlow Notebooks:
Each implementation has both a Python script (.py) and a corresponding Jupyter notebook (.ipynb), with their own README files:
- `ANN Implementation/ann.ipynb` - Interactive notebook for customer churn prediction using ANN
- `Hidden Layer Analysis/Hidden Layers And Hidden Neurons.ipynb` - Analysis of neural network architectures
- `Hyperparameter Tuning/Hyperparameter.ipynb` - Interactive notebook for hyperparameter tuning experiments

### Using Notebooks in Google Colab

1. **Access Google Colab:**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Sign in with your Google account

2. **Upload and Open Notebooks:**
   - Click on `File` > `Upload notebook`
   - Select the `.ipynb` file you want to work with
   - Or use `File` > `Open notebook` > `Upload` to browse your local files

3. **Working with the Notebooks:**
   - Make sure to run the setup cells at the beginning of each notebook
   - For notebooks requiring external data files, you'll need to upload them to Colab's runtime
   - Use the `Runtime` menu to:
     - Run all cells
     - Restart runtime
     - Change runtime type (if you need GPU acceleration)

4. **Save Your Work:**
   - Save a copy to your Google Drive: `File` > `Save a copy in Drive`
   - Download locally: `File` > `Download` > `Download .ipynb`

## Google Colab Hardware Acceleration

Google Colab provides different hardware accelerators to speed up model training:

### Types of Hardware Accelerators

1. **CPU (Central Processing Unit)**
   - Default option
   - Suitable for basic computations and small models
   - Slowest among the three options
   - Use when:
     - Running basic data preprocessing
     - Training very small models
     - Testing code functionality

2. **GPU (Graphics Processing Unit)**
   - Significantly faster than CPU for deep learning
   - Ideal for most deep learning tasks
   - NVIDIA GPUs (Tesla T4, P100, or V100)
   - Use when:
     - Training neural networks
     - Processing image/video data
     - Running parallel computations
   - Free tier limitations apply

3. **TPU (Tensor Processing Unit)**
   - Google's custom-designed AI accelerator
   - Fastest option for specific models
   - Best for TensorFlow models
   - Use when:
     - Training large TensorFlow models
     - Need maximum performance
     - Working with distributed training
   - May require code modifications

### How to Enable Hardware Acceleration in Colab

1. Click on "Runtime" in the top menu
2. Select "Change runtime type"
3. Choose your hardware accelerator:
   - None (CPU)
   - GPU
   - TPU
4. Click "Save"

### Verifying GPU/TPU Connection

For GPU:
```python
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Or for PyTorch
import torch
print("GPU Available: ", torch.cuda.is_available())
print("GPU Device Name: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

For TPU:
```python
import tensorflow as tf
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
    print("No TPU detected")
```

### Best Practices

1. **Resource Management**:
   - Free tier has usage limits
   - Sessions disconnect after 12 hours
   - Save your work frequently
   - Use `runtime.reset()` to clear memory

2. **Choosing Accelerator**:
   - CPU: Data preprocessing, small models
   - GPU: Most deep learning tasks, PyTorch models
   - TPU: Large TensorFlow models, distributed training

3. **Memory Usage**:
   - Monitor memory usage (RAM)
   - Clear output cells when not needed
   - Restart runtime if memory issues occur
   - Use appropriate batch sizes

4. **Performance Tips**:
   - Keep data on Google Drive for faster access
   - Use efficient data loading methods
   - Enable mixed precision training when possible
   - Monitor training with tensorboard

## Note
- The notebooks are self-contained and include all necessary package installations
- Some notebooks may require additional data files - check the notebook contents for specific requirements
- Make sure to select GPU runtime in Colab for notebooks involving deep learning models
