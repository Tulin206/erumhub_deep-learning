import torch
import numpy as np
import matplotlib.pyplot as plt

# From local helper files
from helper_evaluation import set_all_seeds, set_deterministic
from helper_train import train_model
from helper_plotting import plot_training_loss, plot_accuracy, show_examples
from helper_dataset import get_dataloaders_mnist

##########################
### SETTINGS
##########################

RANDOM_SEED = 123
BATCH_SIZE = 256
NUM_HIDDEN_1 = 75
NUM_HIDDEN_2 = 45
NUM_EPOCHS = 100
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Ensure reproducibility: Makes weight init, shuffles, etc. reproducible.
set_all_seeds(RANDOM_SEED)
set_deterministic()

##########################
### MNIST DATASET
##########################

train_loader, valid_loader, test_loader = get_dataloaders_mnist(
    batch_size=BATCH_SIZE,
    validation_fraction=0.1)

# Checking the dataset
for images, labels in train_loader:
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    print('Class labels of 10 examples:', labels[:10])
    break


class MultilayerPerceptron(torch.nn.Module):          # In PyTorch, every model is a subclass of torch.nn.Module.

    """ Constructor:
                    # An input layer (num_features),
                    # Two hidden layers (num_hidden_1, num_hidden_2),
                    # An output layer (num_classes)
    num_features → number of input features (e.g. 28×28 = 784 for MNIST).
    num_classes → number of output classes (e.g. 10 digits).
    drop_proba → dropout probability (but here they hard-coded 0.5 and 0.3).
    num_hidden_1 and num_hidden_2 → sizes of the first and second hidden layers."""

    def __init__(self, num_features, num_classes, drop_proba,
                 num_hidden_1, num_hidden_2):
        super().__init__()

        self.my_network = torch.nn.Sequential(

            # 1st hidden layer
            # Converts each input image (shape [batch_size, 1, 28, 28]) into a flat vector [batch_size, 784].
            torch.nn.Flatten(),

            # Fully connected layer (linear transformation): y = xW^T + b
            # fully connected layer → from input → first hidden layer.
            torch.nn.Linear(num_features, num_hidden_1, bias=False),

            # normalizes activations to help stable learning.
            torch.nn.BatchNorm1d(num_hidden_1),

            # activation function (introduces nonlinearity).
            torch.nn.ReLU(),

            # randomly sets some activations (50% of neurons during training) to zero (helps prevent overfitting).
            torch.nn.Dropout(0.5),

            # 2nd hidden layer
            torch.nn.Linear(num_hidden_1, num_hidden_2, bias=False),  # going from hidden_1 → hidden_2
            torch.nn.BatchNorm1d(num_hidden_2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            # output layer
            torch.nn.Linear(num_hidden_2, num_classes) # going from hidden_2 → output layer (10 classes)
        )

    def forward(self, x):
        logits = self.my_network(x)
        return logits

torch.manual_seed(RANDOM_SEED)
model = MultilayerPerceptron(num_features=28*28,
                             num_hidden_1=NUM_HIDDEN_1,
                             num_hidden_2=NUM_HIDDEN_2,
                             drop_proba=0.5,
                             num_classes=10)
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

"""minibatch_loss_list: per-batch loss values during training, train_acc_list: epoch-wise training accuracy (%), valid_acc_list: epoch-wise validation accuracy (%)"""
minibatch_loss_list, train_acc_list, valid_acc_list = train_model(
    model=model,
    num_epochs=NUM_EPOCHS,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    device=DEVICE,
    logging_interval=100)   # Print training progress every 100 mini-batches. Example: if there are 600 batches per epoch, you’ll see 6 log messages per epoch instead of 600.

plot_training_loss(minibatch_loss_list=minibatch_loss_list,
                   num_epochs=NUM_EPOCHS,
                   iter_per_epoch=len(train_loader),
                   results_dir=None,
                   averaging_iterations=20)   # Smooth the training loss curve by averaging over 20 iterations
plt.show()

plot_accuracy(train_acc_list=train_acc_list,
              valid_acc_list=valid_acc_list,
              results_dir=None)
plt.ylim([80, 100])
plt.show()