# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Check TensorFlow version
print(tf.__version__)

# Load the Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

# Split the dataset into training and testing sets
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Pick one image and add channel + batch dims: (28,28) -> (1,28,28,1)
img = train_images[0].astype("float32")
x = np.expand_dims(np.expand_dims(img, axis=-1), axis=0)

# 3) Define augmentations
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=10,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generate and visualize a few samples
gen = datagen.flow(x, batch_size=1, seed=123)
plt.figure(figsize=(12,3))
for i in range(8):
    aug = next(gen)[0].squeeze()          # (28,28)
    plt.subplot(1,8,i+1)
    plt.xticks([]); plt.yticks([]); plt.grid(False)
    plt.imshow(aug, cmap="gray")
plt.suptitle("ImageDataGenerator: random augmentations of one Fashion-MNIST image")
plt.show()

# Class names for the dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Explore the dataset
print("shape of training image", train_images.shape)            # (60000, 28, 28)
print("length of training image", len(train_labels))            # 60000
print("integer labels of training image", train_labels)         # integers 0–9
print("shape of test image", test_images.shape)                 # (10000, 28, 28)
print("length of test image", len(test_labels))                 # 10000

# Display the first image in the training set
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Preprocess the data by normalizing pixel values to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Display the first 25 images from the training set with their class names
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Build the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Turn logits into probabilities
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# Make predictions
predictions = probability_model.predict(test_images)

# View the prediction probabilities for the first test image
print("\nprobabilities for each of the 10 classes for first test image:\n", predictions[0])

# The predicted label is the one with the highest probability
print("largest probability index for first test image ", np.argmax(predictions[0]))

print("integer class of the first test image", test_labels[0])

def plot_image(i, predictions_array, true_label, img):
  """
  Plots a single image from the dataset along with its predicted and true label.
  The label is colored blue if the prediction is correct, red otherwise.
  """
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  """
  Plots a bar chart of the predicted probabilities for each class for a single image.
  The predicted label is colored red, and the true label is colored blue.
  """
  true_label = true_label[i]                            # 1) pick the true class index for image i
  plt.grid(False)                                       # 2) tidy up the plot
  plt.xticks(range(10), class_names, rotation=45)       # x-axis will show ticks 0..9 (class indices)
  plt.yticks([])

  thisplot = plt.bar(range(10),                         # 3) draw 10 bars (one per class)
                     predictions_array,                 # heights = probabilities for each class (length-10)
                     color="#777777")                   # start all bars in gray

  plt.ylim([0, 1])                                      # 4) probabilities go from 0 to 1
  predicted_label = np.argmax(predictions_array)        # 5) class with highest probability

  thisplot[predicted_label].set_color('red')            # 6) color the predicted class bar red
  thisplot[true_label].set_color('blue')                # 7) color the true class bar blue

for i in [0, 12]:
    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images) # Pass the entire test_labels array
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# Grab an image from the test dataset.
img = test_images[1]
print("shape of first test image", img.shape)
img = (np.expand_dims(img, 0))      # Add a batch dimension
print("test image shape after adding batch diemnsion", img.shape)     # (1, 28, 28)
predictions_single = probability_model.predict(img)       # Get probabilities for that single image
print("\nall probability values for that test image\n", predictions_single)           # 1×10 probs across classes

# Plot the probability bars
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

# Finds the index of the largest probability (the predicted class)
pred_idx = np.argmax(predictions_single[0])
print("print the probability index:", pred_idx, "and the class name:", class_names[pred_idx])