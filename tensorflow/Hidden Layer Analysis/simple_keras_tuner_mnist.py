# Import required libraries
import tensorflow as tf
from tensorflow import keras
import keras_tuner
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load MNIST data
print("Loading MNIST data...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data
# Reshape and normalize the images
X_train = X_train.reshape(-1, 28*28).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28*28).astype('float32') / 255.0

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

def build_model(hp):
    """
    This function builds a model with tunable parameters:
    1. Number of hidden layers (1-3)
    2. Number of neurons in each layer (32-512)
    """
    model = keras.Sequential()

    # Input layer (flatten 28x28 images)
    model.add(keras.layers.Input(shape=(784,)))  # 28*28 = 784

    # Tune number of hidden layers (between 1 and 3)
    n_layers = hp.Int('num_layers', min_value=1, max_value=3)

    # Add hidden layers with tunable number of neurons
    for i in range(n_layers):
        # Number of neurons in this layer (32 to 512)
        units = hp.Int(
            f'units_layer_{i}',
            min_value=32,
            max_value=512,
            step=32
        )

        # Add Dense layer with the tuned number of neurons
        model.add(keras.layers.Dense(
            units=units,
            activation='relu'
        ))

        # Add Dropout for regularization
        dropout_rate = hp.Float(
            f'dropout_{i}',
            min_value=0.0,
            max_value=0.5,
            step=0.1
        )
        model.add(keras.layers.Dropout(dropout_rate))

    # Output layer (10 neurons for digits 0-9)
    model.add(keras.layers.Dense(10, activation='softmax'))

    # Tune learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Create a tuner
print("Creating tuner...")
tuner = keras_tuner.RandomSearch(
    build_model,
    objective='val_accuracy',  # Changed to accuracy for classification
    max_trials=5,
    directory='keras_tuner',
    project_name='mnist_tuning'
)

# Show search space summary
print("\nSearch space summary:")
tuner.search_space_summary()

# Add early stopping
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )
]

# Start the search
print("\nStarting the search...")
tuner.search(
    X_train, y_train,
    epochs=10,  # Reduced epochs for MNIST
    batch_size=128,  # Increased batch size for faster training
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print the results
print("\nBest hyperparameters found:")
print(f"Number of hidden layers: {best_hps.get('num_layers')}")
for i in range(best_hps.get('num_layers')):
    print(f"Layer {i+1}:")
    print(f"  Units: {best_hps.get(f'units_layer_{i}')}")
    print(f"  Dropout rate: {best_hps.get(f'dropout_{i}')}")
print(f"Learning rate: {best_hps.get('learning_rate')}")

# Build the model with the best hyperparameters
best_model = build_model(best_hps)

# Train the model
print("\nTraining the final model with best hyperparameters...")
history = best_model.fit(
    X_train, y_train,
    epochs=20,  # Train for more epochs on final model
    batch_size=128,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Evaluate the model
print("\nEvaluating the model...")
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Save the model
model_path = 'best_mnist_model.keras'
best_model.save(model_path)
print(f"\nModel saved as '{model_path}'")
