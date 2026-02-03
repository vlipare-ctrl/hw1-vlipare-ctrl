#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential
from keras.utils import load_img
from keras.utils import img_to_array

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image


print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


## 

def build_model1():
  model = Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(128, activation=tf.nn.leaky_relu),
    layers.Dense(128, activation=tf.nn.leaky_relu),
    layers.Dense(128, activation=tf.nn.leaky_relu),
    layers.Dense(10)
  ])

  model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
  )

  return model


def build_model2():
  model = Sequential([
    # First Conv Block
    layers.Conv2D(32, (3,3), strides=2, padding='same', activation='relu', input_shape=(32,32,3)),
    layers.BatchNormalization(),

    # Second Conv Block
    layers.Conv2D(64, (3,3), strides=2, padding='same', activation='relu'),
    layers.BatchNormalization(),

    # Third Conv Block
    layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),

    # Fourth Conv Block
    layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),

    # Fifth Conv Block
    layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),

    # Sixth Conv Block
    layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),

    # Flatten and Dense
    layers.Flatten(),
    layers.Dense(10)
  ])

  # Compile the model
  model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
  )

  return model

def build_model3():
  model = Sequential([
    # First Conv Block: normal Conv2D
    layers.SeparableConv2D(32, (3,3), strides=2, padding='same', activation='relu', input_shape=(32,32,3)),
    layers.BatchNormalization(),

    # Second Conv Block: SeparableConv2D
    layers.SeparableConv2D(64, (3,3), strides=2, padding='same', activation='relu'),
    layers.BatchNormalization(),

    # Third Conv Block: SeparableConv2D
    layers.SeparableConv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),

    # Fourth Conv Block: SeparableConv2D
    layers.SeparableConv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),

    # Fifth Conv Block: SeparableConv2D
    layers.SeparableConv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),

    # Sixth Conv Block: SeparableConv2D
    layers.SeparableConv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),

    # Flatten + Dense
    layers.Flatten(),
    layers.Dense(10)
  ])

  # Compile the model
  model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
  )

  return model

def build_model50k():
  model = Sequential([
    # First Conv Block: normal Conv2D
    layers.SeparableConv2D(32, (3,3), strides=2, padding='same', activation='relu', input_shape=(32,32,3)),
    layers.BatchNormalization(),

    # Second Conv Block: SeparableConv2D
    layers.SeparableConv2D(64, (3,3), strides=2, padding='same', activation='relu'),
    layers.BatchNormalization(),

    # Third Conv Block: SeparableConv2D
    layers.SeparableConv2D(96, (3,3), strides=2, padding='same', activation='relu'),
    layers.BatchNormalization(),

    # Flatten + Dense
    layers.Flatten(),
    layers.Dense(10)
  ])

  # Compile the model
  model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
  )

  return model

model1 = build_model1()
model2 = build_model2()
model3 = build_model3()
model50k = build_model50k()

# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
  
  # Normalize
  train_images = train_images / 255.0
  test_images = test_images / 255.0

  # Split training → training + validation (80/20)
  total_images = train_images.shape[0]

  val_images = train_images[int(0.8 * total_images):]
  val_labels = train_labels[int(0.8 * total_images):]

  train_images = train_images[:int(0.8 * total_images)]
  train_labels = train_labels[:int(0.8 * total_images)]

  ########################################
  ## Build and train model 1
  model1.summary()
  model1.fit(train_images, train_labels, epochs=30, validation_data=(val_images, val_labels))
  test_loss, test_acc = model1.evaluate(test_images, test_labels)
  print(f"Model 1 Test accuracy: {test_acc}")

  ## Build, compile, and train model 2 (DS Convolutions)
  model2.summary()
  model2.fit(train_images, train_labels, epochs=30, validation_data=(val_images, val_labels))
  test_loss, test_acc = model2.evaluate(test_images, test_labels)
  print(f"Model 2 Test accuracy: {test_acc}")

  
  ### Repeat for model 3 and your best sub-50k params model
  model3.summary()
  model3.fit(train_images, train_labels, epochs=30, validation_data=(val_images, val_labels))
  test_loss, test_acc = model3.evaluate(test_images, test_labels)
  print(f"Model 3 Test accuracy: {test_acc}")

  ### Repeat for 50k model
  model50k.summary()
  model50k.fit(train_images, train_labels, epochs=30, validation_data=(val_images, val_labels))
  test_loss, test_acc = model50k.evaluate(test_images, test_labels)
  print(f"Model 50k Test accuracy: {test_acc}")
  model50k.save("best_model.h5")


  ## Load Test Image and make prediction
  # Load your test image
  test_img = np.array(load_img(
    './test_image_dog.jpg',
    color_mode='rgb',
    target_size=(32,32)))
  test_img = test_img / 255.0
  test_img = np.expand_dims(test_img, axis=0)

  # Make predictions using one of your trained models (model1, model2, etc.)
  logits = model2.predict(test_img)
  pred_class = np.argmax(logits, axis=1)[0]

  # CIFAR-10 class names
  class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

  print(f"Predicted class: {class_names[pred_class]}")
