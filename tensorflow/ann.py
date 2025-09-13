# Artificial Neural Network


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
print("Independent features:\n", X)
print("Shape of Independent features data:\n", X.shape)
y = dataset.iloc[:, 13]
print("Dependent features:\n", y)
print("Shape of Dependent features data:\n", y.shape)

#Create dummy variables
geography=pd.get_dummies(X["Geography"],drop_first=True)
print("Shape of Geography data:\n", geography.shape)
print("geography data:\n", geography)
gender=pd.get_dummies(X['Gender'],drop_first=True)
print("Shape of gender data:\n", gender.shape)
print("gender data:\n", gender)

## Concatenate the Data Frames
X=pd.concat([X,geography,gender],axis=1)
print("Shape of X:\n", X.shape)
print("Concatenated data:\n", X.values)

## Drop Unnecessary columns
X=X.drop(['Geography','Gender'],axis=1)
print("Shape of X:\n", X.shape)
print("Concatenated data after dropping unnecessary columns:\n", X.values)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print("Shape of X_train:\n", X_train.shape)
print("Shape of X_test:\n", X_test.shape)
print("Shape of y_train:\n", y_train.shape)
print("Shape of y_test:\n", y_test.shape)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# classifier.add(Dense(output_dim = 6, init = 'he_uniform',activation='relu',input_dim = 11))     ## older version
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu',input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, epochs=100, verbose=1)      ## earlier version: nb_epoch=100
print("model history:\n", model_history)
# list all data in history

print(model_history.history.keys())
# summarize history for accuracy
plt.plot(model_history.history['accuracy'])             ## old version: plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_accuracy'])         ## old version: plt.plot(model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)
print("Accuracy:", score)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (Accuracy = {score:.2f})')
plt.show()