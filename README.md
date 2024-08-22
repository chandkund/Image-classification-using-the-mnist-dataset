# Image-classification-using-the-mnist-dataset

This project focuses on building and training a model to classify handwritten digits using the MNIST dataset. The MNIST dataset consists of 60,000 training images and 10,000 test images, each representing a digit from 0 to 9.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Model Evaluation](#model-evaluation)
- [License](#license)

## Project Overview

The goal of this project is to develop a machine learning model that can accurately classify handwritten digits. This is a common benchmark problem in machine learning, especially in the field of deep learning. The project uses a simple convolutional neural network (CNN) to achieve high accuracy on the test set.

## Installation

To run this project, you will need Python along with the following libraries:

- `tensorflow`
- `keras`
- `numpy`
- `matplotlib`

You can install the required packages using `pip`:

```bash
pip install tensorflow keras numpy matplotlib
```

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/chandkund/Image-classification-using-the-mnist-dataset.git
    cd Image-classification-using-the-mnist-dataset
    ```

2. Run the script to train and evaluate the model:

    ```bash
    python train_model.py
    ```

## Code Explanation

- **Import Libraries**:

    ```python
  import tensorflow as tf
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  from tensorflow.keras import models
  import tensorflow_datasets as tfds

    ```

- **Load and Preprocess Data**:

    ```python
    
  # Load the MNIST dataset
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

  # Check the shape of the training data
  print("Training data shape:", x_train.shape)
  print("Training labels shape:", y_train.shape)

  x_train,x_test = x_train/255.0,x_test/255.0

    ```

 - **Visualize  **:
  ```python
  # Display the first 25 images from the training set and their class names
  class_names=['0','1','2','3','4','5','6','7','8','9']
  import matplotlib.pyplot as plt
  plt.figure(figsize=(10,10))
  for i in range(25):
     plt.subplot(5,5,i+1)
     plt.xticks([])
     plt.yticks([])
     plt.grid(False)
     plt.imshow(x_train[i], cmap=plt.cm.binary)
     plt.xlabel(class_names[y_train[i]])
  plt.show()
```


- **Build the CNN Model**:

    ```python
  hidden_layer= 64
  output_layer=10
  model = models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
  ])
    ```

- **Compile and Train the Model**:

    ```python
     model.compile(optimizer = 'adam', loss ='categorical_crossentropy',metrics=['accuracy'])

    history=model.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_test,y_test),verbose=2)
    ```

- **Evaluate the Model**:

    ```python
  test_loss,test_accuracy=model.evaluate(x_test,y_test)    print(f'Test Accuracy: {test_acc}')
  print('Test loss:{0:2f}.Test accuracy:{1:.2f}%'.format(test_loss,test_accuracy*100))
    
    ```

- **Visualize Training History**:

    ```python
    # Assuming class names are the digits from 0 to 9
    class_names=['0','1','2','3','4','5','6','7','8','9']

  # Make predictions on the test set
  predictions = model.predict(x_test)

  # Display the first 5 predictions and actual labels
  for i in range(5):
    predicted_label = tf.argmax(predictions[i]).numpy()
    actual_label = tf.argmax(y_test[i]).numpy()
    print(f"Predicted: {class_names[predicted_label]}, Actual: {class_names[actual_label]}"

    ```

## Model Evaluation

After training, the model is evaluated on the test dataset to determine its accuracy in classifying handwritten digits. The evaluation metrics include accuracy and loss.
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
