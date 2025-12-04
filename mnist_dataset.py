import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from keras.datasets import mnist
object=mnist
(train_img,train_lab),(test_img,test_lab)=object.load_data()
for i in range(20):
  plt.subplot(4,5,i+1)
  plt.imshow(train_img[i],cmap="gray_r")
  plt.title("Digit :{}".format(train_lab[i]))
  plt.subplots_adjust(hspace=0.5)
  plt.axis("off")
print('Training image shape:',train_img.shape)
print('Test image shape:',test_img.shape)
plt.hist(train_img[0].reshape(784),facecolor='blue')
plt.title("pixel vs its intensity",fontsize=16)
plt.ylabel('PIXEL')
plt.xlabel('INTENSITY')
train_img=train_img/255.0
test_img=test_img/255.0
plt.hist(train_img[0].reshape(784),facecolor='blue')
plt.title("pixel vs its intensity",fontsize=16)
plt.ylabel('PIXEL')
plt.xlabel('INTENSITY')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
model = Sequential()
input_layer = Flatten(input_shape=(28, 28))
model.add(input_layer)
hidden_layer1 = Dense(512, activation='relu')
model.add(hidden_layer1)
hidden_layer2 = Dense(512, activation='relu')
model.add(hidden_layer2)
output_layer = Dense(10, activation='softmax')
model.add(output_layer)
model.save('mlp.h5')
loss_and_acc = model.evaluate(test_img, test_lab,verbose=2)
print("Test Loss", loss_and_acc[0])
print("Test Accuracy", loss_and_acc[1])
plt.imshow(test_img[6],cmap="gray_r")
plt.title("Digit :{}".format(test_lab[6]))
prediction=model.predict(test_img)
plt.axis("off")
print('prediction value:',np.argmax(prediction[6]))
if(test_lab[6]==(np.argmax(prediction[6]))):
  print('successful prediction')
else:
  print('unsuccessful prediction')
