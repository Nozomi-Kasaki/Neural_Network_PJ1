import gzip
import os
import pickle
from struct import unpack

import numpy as np

import mynn as nn


# ------------------------------
# Evaluation config
# ------------------------------
MODEL_TYPE = 'CNN'   # choose from {'MLP', 'CNN'}
MODEL_PATH = rf'.\best_models_{MODEL_TYPE.lower()}\best_model.pickle'
VALID_SIZE = 10000
IDX_PATH = r'.\idx.pickle'

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'
test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'


def build_model(model_type):
    if model_type == 'MLP':
        return nn.models.Model_MLP()
    if model_type == 'CNN':
        return nn.models.Model_CNN(input_shape=None)
    raise ValueError(f'Unsupported MODEL_TYPE: {model_type}')


def load_images_labels(images_path, labels_path):
    with gzip.open(images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

    with gzip.open(labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    images = images.astype(np.float32) / 255.0
    return images, labels, rows, cols


def format_inputs(images, rows, cols, model_type):
    if model_type == 'CNN':
        return images.reshape(-1, 1, rows, cols)
    return images


model = build_model(MODEL_TYPE)
model.load_model(MODEL_PATH)

train_imgs, train_labs, rows, cols = load_images_labels(train_images_path, train_labels_path)
test_imgs, test_labs, _, _ = load_images_labels(test_images_path, test_labels_path)

if os.path.exists(IDX_PATH):
    with open(IDX_PATH, 'rb') as f:
        idx = pickle.load(f)
    train_imgs = train_imgs[idx][VALID_SIZE:]
    train_labs = train_labs[idx][VALID_SIZE:]

train_inputs = format_inputs(train_imgs, rows, cols, MODEL_TYPE)
test_inputs = format_inputs(test_imgs, rows, cols, MODEL_TYPE)

train_logits = model(train_inputs)
test_logits = model(test_inputs)

print(f"Train accuracy: {nn.metric.accuracy(train_logits, train_labs):.6f}")
print(f"Test accuracy: {nn.metric.accuracy(test_logits, test_labs):.6f}")
