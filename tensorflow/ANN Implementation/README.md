# Customer Churn Prediction using Artificial Neural Network

This implementation demonstrates how to build and train an Artificial Neural Network (ANN) using TensorFlow/Keras for predicting customer churn in a bank.

## Dataset: Churn_Modelling.csv

The dataset contains customer information from a bank, including:
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary
- Exited (Target variable: whether the customer left the bank)

## Code Structure

The implementation is divided into three main parts:

### 1. Data Preprocessing
```python
# Key steps:
- Import required libraries (numpy, pandas, matplotlib, seaborn)
- Load and inspect the dataset (X: columns 3-13, y: column 13)
- Create dummy variables for categorical features (Geography and Gender) with drop_first=True
- Concatenate dummy variables with original features
- Drop original categorical columns (Geography and Gender)
- Split data into training (80%) and test (20%) sets with random_state=0
- Apply feature scaling using StandardScaler
```

### 2. Building and Training the ANN
```python
# Required Libraries:
- keras
- Sequential model
- Dense layers
- Additional activation functions (LeakyReLU, PReLU, ELU)
- Dropout (available but not used in current implementation)

# Network Architecture:
- Input Layer: 11 features
- First Hidden Layer: 6 neurons with ReLU activation and he_uniform initialization
- Second Hidden Layer: 6 neurons with ReLU activation and he_uniform initialization
- Output Layer: 1 neuron with Sigmoid activation and glorot_uniform initialization

# Training Configuration:
- Optimizer: Adamax
- Loss Function: Binary Cross-entropy
- Metrics: Accuracy
- Batch Size: 10
- Epochs: 100
- Validation Split: 33%
```

### 3. Model Evaluation and Visualization
The model's performance is evaluated using:
- Training and validation accuracy plots over epochs
- Training and validation loss plots over epochs
- Model history tracking (accuracy and loss metrics)
- Detailed visualization of learning curves

## Key Features

1. **Data Preprocessing**
   - Handles categorical variables using one-hot encoding
   - Implements feature scaling for better model performance
   - Proper train-test split for evaluation

2. **Model Architecture**
   - Uses 'he_uniform' initialization for hidden layers
   - Implements ReLU activation for hidden layers
   - Uses 'glorot_uniform' initialization and sigmoid activation for output layer

3. **Visualization**
   - Training/validation accuracy curves
   - Training/validation loss curves
   - Confusion matrix heatmap with accuracy score

## Usage

**Run the Notebook**
   - Open `ann.ipynb` in Jupyter Notebook or Google Colab
   - Ensure `Churn_Modelling.csv` is in the same directory
   - Run all cells sequentially

## Model Performance Monitoring

The notebook provides several visualizations to monitor model performance:

1. **Accuracy Plot**
   - Shows training vs validation accuracy over epochs
   - Helps identify overfitting/underfitting
   - Located in the "Training History" section

2. **Loss Plot**
   - Displays training vs validation loss over epochs
   - Useful for monitoring convergence
   - Located in the "Training History" section

3. **Confusion Matrix**
   - Visual representation of model predictions
   - Shows True Positives, True Negatives, False Positives, and False Negatives
   - Includes overall accuracy score

## Notes

- The model uses a validation split of 33% during training
- Early stopping is not implemented but could be added for better training control
- The model architecture (6 neurons in hidden layers) was chosen based on the dataset size
- Feature scaling is crucial for this implementation due to varying scales of input features
