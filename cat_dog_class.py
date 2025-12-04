import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset (contains cats and dogs among other classes)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Select only cat (class 3) and dog (class 5) images
cat_dog_indices_train = np.where((y_train == 3) | (y_train == 5))[0]
cat_dog_indices_test = np.where((y_test == 3) | (y_test == 5))[0]

train_images = x_train[cat_dog_indices_train] / 255.0  # Normalize
train_labels = np.where(y_train[cat_dog_indices_train] == 3, 0, 1).astype(np.int32)  # Cat: 0, Dog: 1

test_images = x_test[cat_dog_indices_test] / 255.0  # Normalize
test_labels = np.where(y_test[cat_dog_indices_test] == 3, 0, 1).astype(np.int32)  # Cat: 0, Dog: 1

# Define CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),  # Added dropout for regularization
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluate the model
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Function to predict cat or dog
def predict_cat_or_dog(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(32, 32))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return "Cat" if prediction < 0.5 else "Dog"

# Example usage:
print(predict_cat_or_dog("/content/cat.jpg"))
