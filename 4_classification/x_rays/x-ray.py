import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Correct directory paths
train_dir = 'augmented-data/train'
test_dir = 'augmented-data/test'

# Create an ImageDataGenerator object with rescaling
data_generator = ImageDataGenerator(rescale=1./255)

# Load images from the directory for training
training_iterator = data_generator.flow_from_directory(
    train_dir,  # Directory with training images
    target_size=(128, 128),  # Resize images to 128x128
    batch_size=32,  # Number of images per batch
    class_mode='sparse'  # Use sparse for multi-class classification
)

# Load images from the directory for validation
validation_iterator = data_generator.flow_from_directory(
    test_dir,  # Directory with testing images
    target_size=(128, 128),
    batch_size=32,
    class_mode='sparse'
)

# Define the model
model = Sequential()

# Add layers
model.add(tf.keras.layers.Input(shape=(128, 128, 3)))  # Input shape should match your image size
model.add(Flatten())  # Flatten the 2D image data
model.add(Dense(128, activation='relu'))  # Hidden layer
model.add(Dense(3, activation='softmax'))  # Output layer for 3 classes

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',  # Use sparse categorical loss for sparse labels
    metrics=['accuracy']
)

# Fit the model
history = model.fit(
    training_iterator,  # Training data
    steps_per_epoch=len(training_iterator),  # Number of batches per epoch
    epochs=10,  # Number of epochs to train
    validation_data=validation_iterator,  # Validation data
    validation_steps=len(validation_iterator)  # Number of batches for validation
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_iterator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Step 2: Define the number of epochs
epochs = range(1, len(acc) + 1)

# Plotting training and validation accuracy and loss
fig = plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo-', label='Training accuracy')
plt.plot(epochs, val_acc, 'r*-', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo-', label='Training loss')
plt.plot(epochs, val_loss, 'r*-', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Save the plot as an image
fig.tight_layout()
fig.savefig('static/images/my_plots.png')
