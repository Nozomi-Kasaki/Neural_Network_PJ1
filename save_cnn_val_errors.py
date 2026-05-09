import csv
import gzip
import os
import pickle
from struct import unpack

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn


# ------------------------------
# Error analysis config
# ------------------------------
MODEL_PATH = r'.\best_models_cnn\best_model.pickle'
IDX_PATH = r'.\idx.pickle'
VALID_SIZE = 10000
EVAL_BATCH_SIZE = 256
MAX_GRID_IMAGES = 100
OUTPUT_DIR = r'.\cnn_validation_errors'

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'


def load_train_images_labels():
    with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

    with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels, rows, cols


def load_validation_split(images, labels):
    if not os.path.exists(IDX_PATH):
        raise FileNotFoundError(
            f'{IDX_PATH} not found. Run test_train.py first so validation split is reproducible.'
        )

    with open(IDX_PATH, 'rb') as f:
        idx = pickle.load(f)

    valid_idx = idx[:VALID_SIZE]
    return images[valid_idx], labels[valid_idx], valid_idx


def predict_in_batches(model, images, rows, cols):
    inputs = images.astype(np.float32) / 255.0
    inputs = inputs.reshape(-1, 1, rows, cols)

    logits_list = []
    for start in range(0, inputs.shape[0], EVAL_BATCH_SIZE):
        batch = inputs[start:start + EVAL_BATCH_SIZE]
        logits_list.append(model(batch))

    logits = np.concatenate(logits_list, axis=0)
    preds = np.argmax(logits, axis=1)
    probs = nn.op.softmax(logits)
    confidence = np.max(probs, axis=1)
    return preds, confidence


def save_error_metadata(errors):
    csv_path = os.path.join(OUTPUT_DIR, 'errors.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['rank', 'original_index', 'true_label', 'pred_label', 'confidence']
        )
        writer.writeheader()
        for rank, item in enumerate(errors):
            writer.writerow({
                'rank': rank,
                'original_index': int(item['original_index']),
                'true_label': int(item['true_label']),
                'pred_label': int(item['pred_label']),
                'confidence': f"{float(item['confidence']):.6f}",
            })
    return csv_path


def save_error_images(errors, rows, cols):
    image_dir = os.path.join(OUTPUT_DIR, 'images')
    os.makedirs(image_dir, exist_ok=True)

    for rank, item in enumerate(errors):
        path = os.path.join(
            image_dir,
            f"{rank:04d}_idx{int(item['original_index'])}_true{int(item['true_label'])}_pred{int(item['pred_label'])}.png"
        )
        plt.imsave(path, item['image'].reshape(rows, cols), cmap='gray')

    return image_dir


def save_error_grid(errors, rows, cols):
    if len(errors) == 0:
        return None

    grid_errors = errors[:MAX_GRID_IMAGES]
    cols_count = 10
    rows_count = int(np.ceil(len(grid_errors) / cols_count))
    fig, axes = plt.subplots(rows_count, cols_count, figsize=(cols_count * 1.35, rows_count * 1.6))
    axes = np.array(axes).reshape(-1)

    for ax, item in zip(axes, grid_errors):
        ax.imshow(item['image'].reshape(rows, cols), cmap='gray')
        ax.set_title(
            f"T:{int(item['true_label'])} P:{int(item['pred_label'])}\nC:{float(item['confidence']):.2f}",
            fontsize=8
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(grid_errors):]:
        ax.axis('off')

    fig.suptitle(f'CNN Validation Errors ({len(errors)} total, showing {len(grid_errors)})')
    fig.tight_layout()

    grid_path = os.path.join(OUTPUT_DIR, 'error_grid.png')
    fig.savefig(grid_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return grid_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images, labels, rows, cols = load_train_images_labels()
    valid_images, valid_labels, valid_idx = load_validation_split(images, labels)

    model = nn.models.Model_CNN(input_shape=None)
    model.load_model(MODEL_PATH)

    preds, confidence = predict_in_batches(model, valid_images, rows, cols)
    wrong_mask = preds != valid_labels
    wrong_positions = np.where(wrong_mask)[0]

    errors = []
    for pos in wrong_positions:
        errors.append({
            'image': valid_images[pos],
            'original_index': valid_idx[pos],
            'true_label': valid_labels[pos],
            'pred_label': preds[pos],
            'confidence': confidence[pos],
        })

    csv_path = save_error_metadata(errors)
    image_dir = save_error_images(errors, rows, cols)
    grid_path = save_error_grid(errors, rows, cols)

    accuracy = 1.0 - len(errors) / valid_labels.shape[0]
    print(f'Validation accuracy: {accuracy:.6f}')
    print(f'Error samples: {len(errors)} / {valid_labels.shape[0]}')
    print(f'CSV saved to: {csv_path}')
    print(f'Images saved to: {image_dir}')
    if grid_path is not None:
        print(f'Grid saved to: {grid_path}')


if __name__ == '__main__':
    main()
