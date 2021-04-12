#IMPORTS
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#data is the fashion dataset 

data = keras.datasets.fashion_mnist

#load the data 
(train_images, train_labels), (test_images, test_labels) = data.load_data()

#Give the index names
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

#color
train_images = train_images
test_images = test_images

#Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
    ])

#how to compile
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=1)


prediction = model.predict(test_images)

for i in range(10):
    plt.grid(False)
    plt.imshow(test_images[i])
    plt.xlabel('Actual: ' + class_names[test_labels[i]])
    plt.title('Prediction ' + class_names[np.argmax(prediction[i])])
    plt.show()

print(class_names[np.argmax(prediction[0])])

