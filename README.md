<h1 align="center">👚 Fashion Item Classifier ⚙️</h1>

<p align="center">A fashion item classifier developed with TensorFlow 2.0.0 and Keras.</p>

[![Build with TensorFlow](https://img.shields.io/static/v1?label=Made%20with&message=TensorFlow%202.0.0&color=orange)](https://www.tensorflow.org/) [![Build with Keras](https://img.shields.io/static/v1?label=Build%20with&message=Keras&color=red)](https://keras.io/)

It is based of the popular [`fashion_mnist`](https://github.com/zalandoresearch/fashion-mnist) dataset of Keras:

![MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist/raw/master/doc/img/fashion-mnist-sprite.png)

## 📦 Deployment
It's important install **TensorFlow 2.0.0** with PIP using:
```
pip install tensorflow==2.0.0-alpha0
```

Then install other libraries like:
* Pandas
* Numpy
* Matplotlib

With its latest version using PIP.

## 🚀 How it works?

### 📊 Reading data

Load MNIST Fashion-item dataset:

```python
data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

```

The dataset split `train_images` and `test_images` are numpy type array with 784 numbers from 0 to 255. These arrays can be converted into images of 28 by 28 pixels (that's why 784 numbers).

Then reduce the data dividing by 225, beacuse is a good practice shrink our data. Now the values of 225 is equal to 1.0 and so on:

```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

Based on the [labels](https://github.com/zalandoresearch/fashion-mnist#labels) for each fashion item, create a list for ten elements:

```python
class_names = ['T-shirt/top',
               'Trouser',
               'Pullover',
               'Dress',
               'Coat',
               'Sandal',
               'Shirt',
               'Sneaker',
               'Bag',
               'Ankle boot']
```

### 🧠 Neural Network Development

Using `Sequential` class from Keras and add layers.

1. The first layer is flattening input data based on 28x28 pixels of the image.
2. Then, the second layer receive the data from the previous layer and works with it with 128 neurons using RELU activation.
3. The final layer receive the data from the previous layer and works with an ouput of 10 (based on the 10 fashion-item) using SoftMax activation.

```python
model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)),
  keras.layers.Dense(128, activation="relu"),
  keras.layers.Dense(10, activation="softmax")
])
```

### Building the Neural Network

Using `compile` method of `Sequential` using the necessary parameters: 
* **Optimizer:** Set as [`adam`](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c), an optimizing algorithm.
* **Loss:** Set as `sparse_categorical_crossentropy`

```python
model.compile(
  optimizer="adam",
  loss="sparse_categorical_crossentropy",
  metrics=["accuracy"]
)
```
