!pip install tensorflow

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the pretrained VGG16 model (without the top layer for fine-tuning)
vgg_model = VGG16(weights='imagenet', include_top=True)

# 2. Load an image file that you want to classify, resizing it to 224x224 pixels (required by VGG16)
img_path = '/content/cat.jpg'   # Provide the path to your image here
img = image.load_img(img_path, target_size=(224, 224))

# 3. Convert the image to a numpy array and preprocess it for the VGG16 model
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = preprocess_input(img_array)  # Preprocess image for VGG16

# 4. Make predictions
predictions = vgg_model.predict(img_array)

# 5. Decode the predictions to readable labels
from tensorflow.keras.applications.vgg16 import decode_predictions
decoded_predictions = decode_predictions(predictions, top=3)[0]

# 6. Print the top-3 predicted classes
print("Predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i+1}. {label}: {score*100:.2f}%")

# 7. Display the image
plt.imshow(img)
plt.title(f"Predicted: {decoded_predictions[0][1]} ({decoded_predictions[0][2]*100:.2f}%)")
plt.show()
