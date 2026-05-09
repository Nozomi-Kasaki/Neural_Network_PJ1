import math

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn


# ------------------------------
# Visualization config
# ------------------------------
MODEL_TYPE = 'MLP'   # choose from {'MLP', 'CNN'}
MODEL_PATH = rf'.\best_models_{MODEL_TYPE.lower()}\best_model.pickle'


def build_model(model_type):
    if model_type == 'MLP':
        return nn.models.Model_MLP()
    if model_type == 'CNN':
        return nn.models.Model_CNN(input_shape=None)
    raise ValueError(f'Unsupported MODEL_TYPE: {model_type}')


def get_optimizable_layers(model):
    return [layer for layer in model.layers if layer.optimizable]


def visualize_mlp(model):
    layers = get_optimizable_layers(model)
    first_weight = layers[0].params['W']
    last_weight = layers[-1].params['W']

    num_neurons = first_weight.shape[1]
    cols = 10
    rows = math.ceil(num_neurons / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    axes = np.array(axes).reshape(-1)
    for i in range(num_neurons):
        axes[i].imshow(first_weight[:, i].reshape(28, 28), cmap='viridis')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    for i in range(num_neurons, len(axes)):
        axes[i].axis('off')
    fig.suptitle('MLP First-Layer Weights')
    fig.tight_layout()

    plt.figure(figsize=(8, 4))
    plt.imshow(last_weight, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.title('MLP Final Linear Layer')
    plt.xlabel('Class Logit')
    plt.ylabel('Hidden Unit')
    plt.tight_layout()


def visualize_cnn(model):
    layers = get_optimizable_layers(model)
    conv_layers = [layer for layer in layers if layer.params['W'].ndim == 4]
    linear_layers = [layer for layer in layers if layer.params['W'].ndim == 2]

    first_conv = conv_layers[0].params['W']
    num_filters = first_conv.shape[0]
    cols = min(8, num_filters)
    rows = math.ceil(num_filters / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes = np.array(axes).reshape(-1)
    for i in range(num_filters):
        axes[i].imshow(first_conv[i, 0], cmap='viridis')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(f'F{i}', fontsize=9)
    for i in range(num_filters, len(axes)):
        axes[i].axis('off')
    fig.suptitle('CNN First Convolution Filters')
    fig.tight_layout()

    last_linear = linear_layers[-1].params['W']
    plt.figure(figsize=(8, 4))
    plt.imshow(last_linear, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.title('CNN Final Linear Layer')
    plt.xlabel('Class Logit')
    plt.ylabel('Feature Unit')
    plt.tight_layout()


model = build_model(MODEL_TYPE)
model.load_model(MODEL_PATH)

if MODEL_TYPE == 'MLP':
    visualize_mlp(model)
elif MODEL_TYPE == 'CNN':
    visualize_cnn(model)

plt.show()
