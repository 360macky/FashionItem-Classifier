# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Reduce the weight of the images dividing for 225
train_images = train_images / 255.0
test_images = test_images / 255.0
