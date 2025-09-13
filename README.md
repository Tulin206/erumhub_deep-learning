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
- perceptron/code/:
  - `perceptron_pytorch.py` - PyTorch implementation of perceptron
- optimization/code/:
  - `adam.py` - Adam optimizer implementation
  - `scheduler.py` - Learning rate scheduler implementation
  - `sgd_scheduler_momentum.py` - SGD with momentum implementation
  - Helper modules:
    - `helper_dataset.py`
    - `helper_evaluation.py`
    - `helper_plotting.py`
    - `helper_train.py`

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

Available Python Files for Local Development:
- `ann.py` - Artificial Neural Network implementation
- `hidden_layers_&_hidden_neurons.py` - Hidden layers experiments
- `Hyperparameter.py` - Hyperparameter tuning implementation
- `image_classification.py` - Image classification using SVM
- `main.py` - Main project file

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
   - First, copy all files from the pytorch folder to your Google Drive's "Colab Notebooks" folder
   - Then run the following code to set up the correct paths:
     ```python
     import os, sys
     BASE = '/content/drive/MyDrive/Colab Notebooks'  # this can have spaces
     os.chdir(BASE)             # go into the folder
     sys.path.insert(0, BASE)   # add it to Python path
     ```
   - This setup ensures that all notebooks can access helper modules (helper_dataset.py, helper_evaluation.py, helper_plotting.py, helper_train.py)
   - Now you can run notebooks like adam.ipynb, sgd-scheduler-momentum.ipynb, and scheduler.ipynb without path issues

3. **Verify Setup:**
   - In the left sidebar, you should see "MyDrive"
   - Under MyDrive, look for the "Colab Notebooks" folder
   - Make sure all your helper files are visible in this folder

### Available Notebooks

#### PyTorch Notebooks:
- `DNN4HEP_combined_exercise.ipynb`
- perceptron/code/:
  - `perceptron-pytorch.ipynb`
- optimization/code/:
  - `adam.ipynb`
  - `scheduler.ipynb`
  - `sgd-scheduler-momentum.ipynb`

#### TensorFlow Notebooks:
- `Hidden Layers And Hidden Neurons.ipynb`
- `Image Classification Using SVM.ipynb`
- `main.ipynb`

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

## Note
- The notebooks are self-contained and include all necessary package installations
- Some notebooks may require additional data files - check the notebook contents for specific requirements
- Make sure to select GPU runtime in Colab for notebooks involving deep learning models
